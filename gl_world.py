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

Lighting (research/lighting_contract.md): render(lighting=True), or a world with `lighting` set, draws the same
passes into a G-buffer (albedo / normal / material id MRT sharing the depth buffer; every world program writes all
three outputs, and with lighting off the extra two go nowhere, so the unlit bytes are unchanged), runs the effect
plugins of gl_effects.py, then draws the grab lines on top. Fill normals come from the distance to the nearest edge
and the material's height profile; procedural patterns use per-vertex rest-frame coordinates interpolated with mean
value coordinates (world_fill.frag). tests/lit_parity_check.py checks the unlit bytes against the pre-lighting code.

Anything not handled here falls back to pygame: World.draw / draw_grid / Trails.draw keep working on any surface
(the --basic pipeline and the editor tests use them).
"""
import inspect
import os
import weakref
from types import SimpleNamespace

import moderngl
import numpy as np
import pygame

from entities import FILL_EXPAND, Body, EnvConfig
from gl_effects import FrameInfo, make_effects
import looping
import materials
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


def _load(ctx, vert, frag, includes=("world_common.glsl",)):
    """Compile shaders/<vert> (None: world_base.vert, a fullscreen strip) + shaders/<frag> as GLSL 330 core, each file
    of `includes` pasted after the #version line of both (the vertex stage gets #define VERTEX_SHADER first)."""
    common = ""
    for name in includes:
        with open(os.path.join(SHADER_DIR, name)) as f:
            common += f.read() + "\n"
    srcs = []
    for name, define in ((vert or "world_base.vert", "#define VERTEX_SHADER\n"), (frag, "")):
        with open(os.path.join(SHADER_DIR, name)) as f:
            srcs.append("#version 330 core\n" + define + common + f.read())
    return ctx.program(vertex_shader=srcs[0], fragment_shader=srcs[1])


WORLD_INCLUDES = ("world_common.glsl", "material.glsl")


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

    pack() returns dict(fills=(k, 10) int32 [start, count, colour, layer, minx, miny, maxx, maxy, material, winding],
    verts=(m, 2) int32 fill vertices with vert_pt=(m,) their World point indices, lines=(k, 10) int32
    [x1, y1, x2, y2, width, colour, layer, transparency 0..255 (0 opaque), material, kind],
    stamps=(k, 6) int64 [cx, cy, colour, layer, width, material] with radii=(k,) float64).
    material: the id from body_mat (per body; base_mat for ropes and hooks, -1 for the grab lines: overlays drawn
    after the lit effects); kind: 0, +1 / -1 (the body's winding) for the rim lines of the no-outline view, 2 for a
    filled body's black border (no emission). Both only matter with lighting on (gl_world draws the same pixels
    without them)."""

    def __init__(self):
        self.topo = None

    def pack(self, world, filled, zoom, outlines=True, body_mat=None, base_mat=0):
        bodies = world.bodies
        lines, stamps, radii = [], [], []
        fills = np.zeros((0, 10), np.int32)
        verts = np.zeros((0, 2), np.int32)
        vert_pt = np.zeros(0, np.int64)
        nb = len(bodies)
        mat = np.zeros(nb, np.int64) if body_mat is None else np.asarray(body_mat, np.int64)
        if nb:
            if self.topo is None or not self.topo.valid(world):
                self.topo = _Topology(world)
            t = self.topo
            P = world.P * zoom
            body_layer = 2 * np.arange(nb, dtype=np.int64)  # +1 for a segment's inner line
            if filled:
                fill_c = pack_rgb([b.fill_col for b in bodies])
                if t.fill_bodies.size:
                    fills, verts = self._fills(t, P, fill_c, body_layer, lines, bodies, outlines, mat)
                    vert_pt = t.loop_idx
                if t.seg_bodies.size:
                    self._segments(t, P, fill_c, body_layer, bodies, lines, stamps, radii, outlines, mat)
                dots = t.dot
            else:
                dots = np.ones(nb, dtype=bool)  # the points view: Body.draw for every body
            if dots.any():
                self._points(t, P, dots, body_layer, bodies, lines, stamps, radii, mat)
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
                        f2, v2 = self._fills(sub, Ps, fill_c, body_layer, lines, bodies, outlines, mat)
                        f2[:, 0] += len(verts)
                        fills, verts = np.concatenate([fills, f2]), np.concatenate([verts, v2])
                        vert_pt = np.concatenate([vert_pt, sub.loop_idx])
                    if sub.seg_bodies.size:
                        self._segments(sub, Ps, fill_c, body_layer, bodies, lines, stamps, radii, outlines, mat)
                    d = dots & mask
                else:
                    d = mask
                if d.any():
                    self._points(t, Ps, d, body_layer, bodies, lines, stamps, radii, mat)
        layer = 2 * nb
        for r in world.ropes:
            if r.width >= 1:
                a, b = _px(r.end_pos(0) * zoom), _px(r.end_pos(1) * zoom)
                lines.append(np.array([[a[0], a[1], b[0], b[1], r.width, pack_rgb([r.col])[0], layer, 0, base_mat, 0]]))
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
                stamps.append(np.array([[c[0], c[1], cc, layer + k, width, base_mat]]))
                radii.append(np.array([float(rad)]))
            layer += 3
        if world.grabbed is not None and nb:
            a, b = _px(world.P[:, world.grabbed] * zoom), _px(np.asarray(world.grab_target) * zoom)
            lines.append(np.array([[a[0], a[1], b[0], b[1], 1, pack_rgb([(255, 255, 255)])[0], layer, 0, -1, 0]]))
        if getattr(world, "grab_set", None) is not None and nb:  # an area grab: a line per held body
            ca, cb, w = world.grab_lines()  # center to cursor; opacity 0.5 x the body's mean pull weight
            a, b = _px(ca * zoom), _px(cb * zoom)
            k = len(w)
            lines.append(np.stack([a[0], a[1], b[0], b[1], np.ones(k), np.full(k, pack_rgb([(255, 255, 255)])[0]),
                                   np.full(k, layer), np.round(255 * (1 - 0.5 * w)), np.full(k, -1), np.zeros(k)],
                                  axis=1))
        cat = lambda xs, w: np.concatenate(xs).astype(np.int64) if xs else np.zeros((0, w), np.int64)
        lines = cat(lines, 10)
        return dict(fills=fills, verts=verts, vert_pt=vert_pt, lines=lines[lines[:, 4] >= 1].astype(np.int32),
                    stamps=cat(stamps, 6), radii=np.concatenate(radii) if radii else np.zeros(0))

    @staticmethod
    def _fills(t, P, fill_c, body_layer, lines, bodies, outlines=True, mat=None):
        """Closed polygons: draw_filled rounds the loop to whole pixels, fills it, then draw_border (entities.py):
        the 1px outline plus BORDER_WIDTH - 1 copies shifted 1px outward along each edge's minor axis."""
        loop = np.round(P[:, t.loop_idx])  # (2, m) whole pixels, rounded like draw_filled
        loop = np.clip(np.nan_to_num(loop, nan=-COORD_LIMIT), -COORD_LIMIT, COORD_LIMIT).astype(np.int64)
        fb, starts = t.fill_bodies, t.loop_start
        lo = np.minimum.reduceat(loop, starts, axis=1)
        hi = np.maximum.reduceat(loop, starts, axis=1)
        wind_b = np.array([b.winding for b in bodies]).astype(np.int64)
        fills = np.stack([starts, t.loop_len, fill_c[fb], body_layer[fb], lo[0], lo[1], hi[0], hi[1], mat[fb],
                          wind_b[fb]], axis=1)
        nxt = loop[:, t.loop_next]
        d = nxt - loop
        winding = wind_b[t.loop_body]
        steep = np.abs(d[1]) >= np.abs(d[0])
        sx = np.where(steep, np.sign(winding * d[1]), 0).astype(np.int64)
        sy = np.where(steep, 0, np.sign(winding * -d[0])).astype(np.int64)
        col = np.full(loop.shape[1], pack_rgb([BORDER_COL])[0])
        layer = body_layer[t.loop_body]
        ones = np.ones(loop.shape[1], np.int64)
        lm, kind = mat[t.loop_body], 2 * ones  # border lines: flat normals, no emission
        if not outlines:  # no border: the shape grown FILL_EXPAND px outward in its own fill colour
            col = fill_c[t.loop_body]
            kind = winding  # rim lines: normals tilted outward
        for k in range(BORDER_WIDTH if outlines else 1 + FILL_EXPAND):
            lines.append(np.stack([loop[0] + k * sx, loop[1] + k * sy, nxt[0] + k * sx, nxt[1] + k * sy, ones, col,
                                   layer, 0 * ones, lm, kind], axis=1))
        return fills.astype(np.int32), loop.T.astype(np.int32)

    @staticmethod
    def _segments(t, P, fill_c, body_layer, bodies, lines, stamps, radii, outlines=True, mat=None):
        """Two-point bodies in the filled view: a thick border-coloured line with round ends, the fill line on top."""
        sb = t.seg_bodies
        a, b = _px(P[:, t.seg_first]), _px(P[:, t.seg_first + 1])
        sw = np.maximum(np.array([bodies[k].stick_width for k in sb], dtype=np.int64), 1)
        outer = sw + 2 * BORDER_WIDTH
        border = np.full(sb.size, pack_rgb([BORDER_COL])[0])
        layer, zero, m = body_layer[sb], np.zeros(sb.size, np.int64), mat[sb]
        if outlines:
            lines.append(np.stack([a[0], a[1], b[0], b[1], outer, border, layer, zero, m, zero + 2], axis=1))
            for end in (a, b):
                stamps.append(np.stack([end[0], end[1], border, layer, zero, m], axis=1))
                radii.append(outer / 2)
        fw = sw if outlines else sw + 2 * FILL_EXPAND
        lines.append(np.stack([a[0], a[1], b[0], b[1], fw, fill_c[sb], layer + 1, zero, m, zero], axis=1))

    @staticmethod
    def _points(t, P, which, body_layer, bodies, lines, stamps, radii, mat=None):
        """Body.draw for the bodies in `which`: springs (draw_edges) in stick_col, then a disc per point."""
        stick_c = pack_rgb([b.stick_col for b in bodies])
        point_c = pack_rgb([b.point_col for b in bodies])
        sel = which[t.spring_body]
        if sel.any():
            i, j = t.springs[:, sel]
            sb = t.spring_body[sel]
            a, b = _px(P[:, i]), _px(P[:, j])
            sw = np.array([bd.stick_width for bd in bodies], dtype=np.int64)[sb]
            lines.append(np.stack([a[0], a[1], b[0], b[1], sw, stick_c[sb], body_layer[sb], 0 * sw, mat[sb], 0 * sw],
                                  axis=1))
        sel = np.flatnonzero(which[t.point_body])
        if sel.size:
            pb = t.point_body[sel]
            c = _px(P[:, sel])
            zero = np.zeros(sel.size, np.int64)
            stamps.append(np.stack([c[0], c[1], point_c[pb], body_layer[pb], zero, mat[pb]], axis=1))
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




def _body_rest(body, seq):
    """(2, n) rest-frame coordinates (world units) of a body's points for its surface pattern: a fixed body's own
    points (world space, so neighbouring static tiles' patterns line up), a rigid body's `rest`, else its shape now
    (first seen: right after it is added) about its centre. Moving bodies get a per-body offset (golden-ratio
    sequence) so equal shapes don't show the same patch of pattern."""
    pts = np.asarray(body.points, dtype=np.float64)
    if body.fixed:
        return pts.copy()
    rest = getattr(body, "rest", None)
    if rest is None or np.shape(rest) != pts.shape:
        rest = pts - pts.mean(axis=1, keepdims=True) if pts.size else pts
    off = np.array([(seq * 0.6180339887) % 1.0, (seq * 0.7548776662) % 1.0]) * 1000.0
    return np.asarray(rest, dtype=np.float64) + off[:, None]


class GLWorld:
    """The world layer: `tex` (RGBA8, the render surface's size, row 0 = top) after render().

    Lighting on (render(lighting=True) or world.lighting): the same passes draw into the G-buffer `gbuf` (albedo,
    normal, mat: research/lighting_contract.md) instead, the effects (gl_effects.py) turn it into `tex`, and the
    grab lines go on top. Materials: each body's resolved materials.Material gets an id (identical ones share it,
    0 = none, 1 = the default, also used by ropes and hooks), packed into `material_tex` (materials.pack), rebuilt
    only when some body's material / material_over changes. Rest-frame coordinates for the surface patterns are
    kept per body (_body_rest, recorded the first time a body is drawn) and uploaded in `uvs_tex`, parallel to the
    fill vertices."""

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
            "fill": _load(ctx, "world_fill.vert", "world_fill.frag", WORLD_INCLUDES),
            "line": _load(ctx, "world_line.vert", "world_line.frag", WORLD_INCLUDES),
            "stamp": _load(ctx, "world_stamp.vert", "world_stamp.frag", WORLD_INCLUDES),
            "points": _load(ctx, "world_points.vert", "world_points.frag"),
            "copy": _load(ctx, None, "world_copy.frag", ()),
        }
        for prog in self.progs.values():
            if "u_size" in prog:
                prog["u_size"].value = self.size
        self.base_vao = ctx.vertex_array(self.progs["base"], [])
        self.fills = _Stream(ctx, self.progs["fill"], "4i 4i 2i/i", ("in_info", "in_box", "in_mat"))
        self.lines = _Stream(ctx, self.progs["line"], "4i 4i 2i/i", ("in_ends", "in_style", "in_mat"))
        self.stamps = _Stream(ctx, self.progs["stamp"], "4i 4i 3i/i", ("in_pos", "in_mask", "in_style"))
        self.points = _Stream(ctx, self.progs["points"], "2i 4i", ("in_pos", "in_col"))
        self.verts_tex = None
        self.uvs_tex = ctx.texture((1, 1), 2, dtype="f4")
        self.uvs_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
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
        self.progs["fill"]["u_uvs"].value = 1
        self.progs["stamp"]["u_atlas"].value = 0
        for name in ("fill", "line", "stamp"):
            self.progs[name]["u_materials"].value = 2
        self.progs["copy"]["u_src"].value = 0
        # lighting: the G-buffer and the effects are made on the first lit frame
        self.gbuf, self.gbuf_fbo, self.effects, self.frame = None, None, None, None
        self.material_tex = ctx.texture((materials.TEXELS, 1), 4, dtype="f4")
        self.material_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.n_materials, self.body_mat = 1, np.zeros(0, np.int64)
        self._mat_sig, self._mat_bytes = None, None
        self._rest, self._rest_seq = weakref.WeakKeyDictionary(), 0
        self._rest_topo, self._rest_all = None, np.zeros((0, 2), np.float32)
        self._uv_key = None  # (rest array, vert_pt) last uploaded to uvs_tex
        self._fs_vaos = {}
        self.stats = {}

    # ---------------- helpers for effects (gl_effects.FrameInfo)
    def load(self, vert, frag, includes=()):
        return _load(self.ctx, vert, frag, tuple(includes))

    def fullscreen(self, prog):
        vao = self._fs_vaos.get(prog.glo)
        if vao is None:
            vao = self._fs_vaos[prog.glo] = self.ctx.vertex_array(prog, [])
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    def release(self):
        """Release the effects and the G-buffer (the rest goes with the context)."""
        for e in self.effects or ():
            e.release()
        self.effects = None
        for t in (self.gbuf or {}).values():
            t.release()
        if self.gbuf_fbo is not None:
            self.gbuf_fbo.release()
        self.gbuf, self.gbuf_fbo = None, None

    # ---------------- passes
    def _set_clip(self, clip, lit, zoom):
        for prog in self.progs.values():
            if "u_clip" in prog:
                prog["u_clip"].value = clip
            if "u_lit" in prog:
                prog["u_lit"].value = lit
        self.progs["fill"]["u_zoom"].value = float(zoom)

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

    def render(self, world, view_size, filled, zoom, background, grid=None, trails=None, outlines=True,
               lighting=False):
        """Draw the world into self.tex. view_size: the world view's size in px (drawing is clipped to it, like the
        subsurface the CPU path draws into); background: RGB; grid: None (off) or draw_grid colour overrides ({} for
        its defaults); trails: a trails.Trails (drawn when it has pixels) or None; lighting: the lit renderer (also
        on when the world has `lighting` set)."""
        ctx = self.ctx
        prev = ctx.fbo if ctx.fbo is not None else ctx.screen
        lit = bool(lighting or getattr(world, "lighting", False))
        clip = (int(min(view_size[0], self.size[0])), int(min(view_size[1], self.size[1])))
        self._set_clip(clip, lit, zoom)
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        has_trails = self._trails(trails, clip, zoom)
        self.fbo.use()
        self.fbo.clear(depth=1.0)
        if lit:
            self._ensure_gbuf()
            self._materials(world)
            self.gbuf_fbo.use()  # (shares self.depth, just cleared; the base pass writes every pixel)
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
        overlays = self._bodies(world, filled, zoom, outlines, lit)
        if lit:
            self._effects(world, clip, zoom, filled)
            self._overlays(overlays)
        if prev is not None:
            prev.use()

    # ---------------- lighting
    def _ensure_gbuf(self):
        if self.gbuf is not None:
            return
        ctx = self.ctx
        self.gbuf = {"albedo": ctx.texture(self.size, 4), "normal": ctx.texture(self.size, 4),
                     "mat": ctx.texture(self.size, 1, dtype="u2")}
        for t in self.gbuf.values():
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.gbuf_fbo = ctx.framebuffer([self.gbuf["albedo"], self.gbuf["normal"], self.gbuf["mat"]], self.depth)

    def _materials(self, world):
        """Material ids per body (self.body_mat) and the material table, redone only when a body's material or
        material_over changed (or bodies were added / removed): per frame this is one comparison per body."""
        bodies = world.bodies
        sig = [(getattr(b, "material", None), getattr(b, "material_over", None) or None,  # (+ fill_col: of_body's
                tuple(() if getattr(b, "fill_col", None) is None else b.fill_col)) for b in bodies]  # glow hue)
        if sig == self._mat_sig:
            return
        self._mat_sig = [(m, dict(o) if o else None, c) for m, o, c in sig]  # copies: in-place edits show up too
        table = [None, materials.resolve()]
        ids = {table[1]: 1}
        body_mat = np.empty(len(bodies), np.int64)
        for k, b in enumerate(bodies):
            m = materials.of_body(b)
            i = ids.get(m)
            if i is None:
                i = ids[m] = len(table)
                table.append(m)
            body_mat[k] = i
        self.body_mat, self.n_materials = body_mat, len(table)
        packed = np.zeros((len(table), materials.TEXELS, 4), np.float32)
        packed[0] = materials.pack(materials.resolve())  # row 0: the background (lit, default material)
        for i, m in enumerate(table[1:], 1):
            packed[i] = materials.pack(m)
        data = packed.tobytes()
        if data == self._mat_bytes:
            return
        if self.material_tex.size[1] != len(table):
            self.material_tex.release()
            self.material_tex = self.ctx.texture((materials.TEXELS, len(table)), 4, dtype="f4")
            self.material_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.material_tex.write(data)
        self._mat_bytes = data

    def _rest_coords(self, world):
        """(n_points, 2) float32 rest-frame coordinates of every World point (see _body_rest), rebuilt when the
        World repacks. Bodies are recorded the first time they are seen (also with lighting off, so the rest frame
        is the shape they were added with)."""
        topo = self.packer.topo
        if topo is self._rest_topo:
            return self._rest_all
        parts = []
        for b in world.bodies:
            r = self._rest.get(b)
            if r is None or r.shape[1] != b.points.shape[1]:
                r = self._rest[b] = _body_rest(b, self._rest_seq)
                self._rest_seq += 1
            parts.append(r)
        self._rest_all = (np.concatenate(parts, axis=1).T.astype(np.float32) if parts
                          else np.zeros((0, 2), np.float32))
        self._rest_topo = topo
        return self._rest_all

    def _effects(self, world, clip, zoom, filled=True):
        """The lit effects in order (gl_effects); none: the albedo as is."""
        ctx = self.ctx
        if self.effects is None:
            self.effects = make_effects(ctx, self.size)
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        if not self.effects:
            self.fbo.use()
            self.gbuf["albedo"].use(0)
            self.fullscreen(self.progs["copy"])
            return
        period = getattr(world, "period", None)
        if period is not None:
            period = (float(period[0]) * zoom, float(period[1]) * zoom)
        self.frame = FrameInfo(self.size, clip, float(zoom), period, world, self.gbuf, self.material_tex,
                               self.n_materials, {}, gw=self, ctx=ctx, fbo=self.fbo, filled=bool(filled))
        for e in self.effects:
            e.run(self, self.frame)
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)

    def _overlays(self, lines):
        """The grab lines over the lit result (blended, depth-tested against the bodies' layers like unlit)."""
        if not len(lines):
            return
        ctx = self.ctx
        self.fbo.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ZERO, moderngl.ONE
        self.material_tex.use(2)
        n = self.lines.write(lines)
        self.lines.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)

    def _bodies(self, world, filled, zoom, outlines=True, lit=False):
        """Fills, lines and circles into the bound framebuffer; returns the overlay lines left for after the
        effects (lighting on; with it off they are drawn here, in the one line pass, exactly as before)."""
        ctx = self.ctx
        if lit:
            data = self.packer.pack(world, filled, zoom, outlines, self.body_mat, 1)
        else:
            data = self.packer.pack(world, filled, zoom, outlines)
        rest = self._rest_coords(world)
        fills, verts, lines, stamps = data["fills"], data["verts"], data["lines"], data["stamps"]
        self.stats = {"fills": len(fills), "lines": len(lines), "stamps": len(stamps)}
        self.material_tex.use(2)
        overlays = lines[:0]
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        if len(fills):
            rows = -(-len(verts) // VERT_ROW)
            if self.verts_tex is None or self.verts_tex.size[1] < rows:
                cap = max(rows, 2 * self.verts_tex.size[1]) if self.verts_tex is not None else rows
                if self.verts_tex is not None:
                    self.verts_tex.release()
                self.uvs_tex.release()
                self.verts_tex = ctx.texture((VERT_ROW, cap), 2, dtype="i4")
                self.verts_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self.uvs_tex = ctx.texture((VERT_ROW, cap), 2, dtype="f4")
                self.uvs_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self._uv_key = None
            buf = np.zeros((rows * VERT_ROW, 2), np.int32)
            buf[:len(verts)] = verts
            self.verts_tex.write(buf.tobytes(), viewport=(0, 0, VERT_ROW, rows))
            vp = data["vert_pt"]
            key = self._uv_key
            if lit and not (key is not None and key[0] is rest and np.array_equal(key[1], vp)):
                # rest coordinates change only with the topology or the periodic copies drawn
                uv = np.zeros((rows * VERT_ROW, 2), np.float32)
                uv[:len(verts)] = rest[vp]
                self.uvs_tex.write(uv.tobytes(), viewport=(0, 0, VERT_ROW, rows))
                self._uv_key = (rest, vp.copy())
            self.verts_tex.use(0)
            self.uvs_tex.use(1)
            n = self.fills.write(fills)
            self.fills.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        if lit and len(lines):  # G-buffer: opaque lines only (no blending into normals / ids); overlays later
            over = lines[:, 8] < 0
            overlays, lines = lines[over], lines[~over]
        if len(lines):
            n = self.lines.write(lines)
            if not lit:
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
        return overlays

    def _stamps(self, stamps, radii):
        """(k, 11) [cx, cy, offx, offy, atlas x, atlas y, w, h, colour, layer, material] for the circles pygame
        would draw."""
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
        return np.column_stack([s[:, 0], s[:, 1], m[:, 4], m[:, 5], m[:, 0], m[:, 1], m[:, 2], m[:, 3], s[:, 2],
                                s[:, 3], s[:, 5]])
