"""Per-chunk stuckness test of a penrose / diamonds map: how well does one region of the map drain a crowd of shapes?

    python3 chunk_test.py "evolved 1 halloween" [--seeds 3] [--grids grid] [--fill 0.5] [--seconds 5]
                          [--workers 1] [--out results.json]

The map (maps/penrose/NAME.json, or a path) is split into a CHUNKS x CHUNKS (4 x 4) grid of chunk rects; three more
grids shift it by half a chunk in x ("shift_x"), y ("shift_y") and both ("shift_xy"), so their chunks straddle the
first grid's seams (the world loops, so the last row / column of a shifted grid wraps across the world's edge).
chunk_rects lists them all, with ids like "grid r1c2" / "shift_xy r3c0".

One chunk test (ChunkTest, run_chunk):
  * World: the chunk rect plus MARGIN (10%) of its size on every side, holding only the tiles that overlap the chunk
    rect (whole tiles, at their map positions relative to the chunk, across the world's wrap), nothing else: no walls,
    no looping, no neighbouring tiles. Physics is the app's own (solver.World, EnvConfig defaults, which the maps
    don't override; the CPU backend, so it's bit-deterministic), stepped exactly as the app and evolve_layout do:
    World.step(1/60) per frame = 4 fixed ticks of 1/240 s, one CCD span per frame.
  * Shapes: the map's spawn settings (scene.diamonds_spawn: polygon circumradius range, side counts taken in turn,
    pumpkin share and half width range, spread evenly through the order as scene.spawn_kinds does; same body
    constructors and stiffness as scene.build_diamonds_scene) placed at random, non-overlapping, wholly inside the
    chunk rect and clear of its tiles, until their placement circles (circumcircle; a pumpkin's REACH circle) cover
    `fill` (default 0.5) of the chunk's free area (rect area minus the tiles' area inside it). Shapes for the whole
    target are drawn first and placed largest first, each at the snuggest of a few random free spots (SNUG); ones
    that don't fit are skipped, then more are drawn one at a time to top up (see ChunkTest._place). Random placement
    saturates: on evolved 1 halloween 0.5 is always met, 0.6 gives ~0.56-0.58 and 0.7 only ~0.59-0.60, so the
    achieved fill is reported ("fill_achieved"). Each shape gets a random velocity: by default (vel None) drawn
    like the whole map's flow, vx ~ N(0, VEL_STD[0]) and vy ~ N(VEL_MEAN[1], VEL_STD[1]) (VEL_MEAN / VEL_STD:
    measured on evolved 1 halloween, 168 shapes, t = 30..60 s: median speed 177 px/s, 98% moving down); with `vel`
    a number, uniform direction and speed 0..vel (the old test: 30).
  * Run `seconds` (default 4) of sim time. Once per frame, a shape whose bounding box lies wholly outside the chunk
    rect (below it, or off a side) has cleared the chunk: its clear time is recorded and it is removed, so nothing
    piles up and it can't come back.
  * Score:
        score       = cleared shapes / shapes                 (1 = the chunk drained completely, 0 = nothing left)
        area_score  = cleared shape area / shape area          (the same weighted by polygon area)
        flow        = mean over shapes of 1 - t_clear / T      (a shape still there at T counts 0; rewards fast
                                                               draining, so it separates chunks that all clear)
        free        = 1 - slow shapes / shapes                 (slow: still in the chunk at T and slower than
                                                               SLOW_SPEED over the last SLOW_WINDOW s; the
                                                               evolution's metric)
    and the shapes still in the chunk at the end are "stuck": their centres (map coordinates, wrapped) and their
    speed over the last SLOW_WINDOW s go into "stuck" ([x, y, px/s, radius]) for a heatmap.
    The 5 px/s / 0.5 s settings come from a parameter search (settle_collect / settle_analyze, 2177 runs, on the
    old slow starts and all sizes): the slow count at 4-5 s ranked chunks like the count still in at 10 s (Spearman
    0.95-0.98; 0.68-0.87 counting every shape in).

Deterministic for a given (map, chunk, seed, settings): the seed draws the shapes, their placement and velocities.
The CLI runs every chunk of the chosen grids on N seeds (seed = 1..N), prints a per-chunk table (mean, std across
seeds, wall time) and writes JSON. chunk_viewer.py shows the same test in real time.

Other layouts (chunk-based evolution): a layout is a set of tile keys (diamonds_map.tile_key, as a map file stores
them under "tiles") of the map's own tiling. run_chunk_keys / ChunkTest(keys=...) test a chunk of such a layout in the
map's settings (spawn, tiling, shift); run_chunk(info, ...) == run_chunk_keys(info, info.keys, ...) bit for bit.
ChunkPool runs batches of (keys, chunk id, seed) jobs on a persistent process pool; whole_map_score runs one
whole-map evolve_layout scoring run of a layout (a spot check).
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

CHUNKS = 4  # chunks per side of the grid
MARGIN = 0.10  # empty space around the chunk rect, as a share of its size, on every side
FILL = 0.5  # share of the chunk's free area covered by the shapes' placement circles
SECONDS = 5.0  # sim time per test
VEL = None  # None: initial velocities like the whole map's (VEL_MEAN / VEL_STD); a number: speed 0..VEL, any way
VEL_MEAN = (0.0, 173.0)  # px/s (y down): shape velocity on the whole map (evolved 1 halloween, t = 30..60 s)
VEL_STD = (72.0, 115.0)
# the shapes that get stuck most: each kind's size drawn only from one end of its range, this share of it: pumpkins
# and heptagons from the top (the largest), hexagons from the bottom (the smallest). None: the whole ranges.
EXTREMES = 1.0 / 3.0
SLOW_SPEED = 5.0  # px/s: a remaining shape slower than this over the last SLOW_WINDOW is "slow" (settled: stuck)
SLOW_WINDOW = 0.5  # s: the speed is the centroid's displacement over this window / the window
GAP = 1.0  # world units kept between placement circles, and between a circle and a tile
TRIES = 2048  # random positions tried per shape (in batches of CANDIDATES) before it is skipped
CANDIDATES = 256
# each shape goes to the snuggest (least clearance) of the first SNUG valid random positions: 1 is plain random
# sequential placement, which jams at ~57% circle coverage here; a few more pack denser while staying spread out
SNUG = 16
FIELD_SLACK = 0.75  # the tile distance field's sampling error bound (half a sample diagonal, 0.707)
MAX_MISSES = 12  # placement's top-up stops after this many shapes in a row found no spot (see ChunkTest._place)
FPS = 60  # World.step(1 / FPS) per frame (4 ticks of 1/240 s, one CCD span)
GRIDS = {"grid": (0.0, 0.0), "shift_x": (0.5, 0.0), "shift_y": (0.0, 0.5), "shift_xy": (0.5, 0.5)}


def map_path(name):
    """A map name in maps/penrose (or maps/diamonds), or a path to a layout file."""
    if os.path.isfile(name):
        return name
    for folder in ("penrose", "diamonds"):
        p = os.path.join(REPO, "maps", folder, name + ".json")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(name)


class MapInfo:
    """What a chunk test needs from a map: world size, tile polygons (as scene.build_diamonds_scene places them),
    their palette picks, spawn settings and lighting block (for the viewer's colours)."""

    def __init__(self, name):
        import scene
        from diamonds_map import layout_lighting, layout_shift, load_layout
        self.path = map_path(name)
        self.name = os.path.splitext(os.path.basename(self.path))[0]
        with open(self.path) as f:
            data = json.load(f)
        self.size = tuple(float(v) for v in data.get("size", (1800.0, 1169.0)))
        W, H = self.size
        # the real map's own builder, so the tiles are exactly the app's (its shapes are ignored)
        world = scene.build_diamonds_scene(W, H, layout=self.path)
        tiles = [b for b in world.bodies if getattr(b, "palette_role", None) == "diamond"]
        self.tiles = [np.array(b.points, dtype=float) for b in tiles]
        self.tile_picks = [int(getattr(b, "palette_pick", 0)) for b in tiles]
        self.spawn = scene.diamonds_spawn(self.path)
        self.lighting = layout_lighting(self.path)
        self.palette = self.lighting.get("palette") or getattr(world, "palette", None)
        self.tiling = scene._layout_tiling(self.path)
        self.shift = layout_shift(self.path)
        self.keys = frozenset(norm_key(k) for k in load_layout(self.path)[0])  # the map's own layout
        self._index = None  # tile key -> (tiling order, corners): built on first use (tile_index)
        self._layouts = {}  # layout_tiles' cache: frozenset of keys -> its tiles

    def tile_index(self):
        """{tile key: (tiling order, corners (2, n))} of every tile of the map's tiling lying in the world: the
        candidates a layout picks from, as scene.build_diamonds_scene / diamonds_map.pick_diamonds build them."""
        if self._index is None:
            import scene
            from diamonds_map import lattice_diamonds, penrose_tiles, steep_tiles
            W, H = self.size
            t = self.tiling
            if t["tiling"] == "penrose":
                side = float(t["edge"])
                tiles = steep_tiles(penrose_tiles(W, H, side, seed=int(t["pentagrid_seed"]),
                                                  rotation=float(t["rotation"])), float(t["min_slope"]))
            else:
                tiles = lattice_diamonds(W, H, scene.DIAMOND_SIDE, seed=scene.DIAMONDS_SEED, y0=scene.DIAMOND_SIDE)
            eps = 1e-6  # pick_diamonds' "inside" test and key_index (the same reductions)
            inside = [d[0] for d in tiles if d[0][0].min() >= -eps and d[0][0].max() <= W + eps
                      and d[0][1].min() >= -eps and d[0][1].max() <= H + eps]
            self._index = {}
            for k, c in enumerate(inside):
                mx, my = np.asarray(c, dtype=float).mean(axis=1).tolist()
                self._index[(round(mx, 1), round(my, 1))] = (k, c)
        return self._index

    def layout_tiles(self, keys):
        """[(polygon (2, n) in map coordinates, palette pick)] of a layout's tiles, exactly as
        scene.build_diamonds_scene builds the map with them: tiling order, the map's shift, palette picks drawn in
        that order. KeyError if a key isn't a tile of the map's tiling (lying in the world)."""
        keys = keys if isinstance(keys, frozenset) else frozenset(norm_key(k) for k in keys)
        got = self._layouts.get(keys)
        if got is not None:
            return got
        import scene
        from diamonds_map import shift_diamonds
        index = self.tile_index()
        missing = [k for k in keys if k not in index]
        if missing:
            raise KeyError(f"{len(missing)} layout keys aren't tiles of {self.name}'s tiling, e.g. {missing[:3]}")
        picked = sorted(index[k][0] for k in keys)
        corners = {k: c for k, c in index.values()}
        diamonds = shift_diamonds([(corners[i], None) for i in picked], self.shift, *self.size)
        rng = np.random.default_rng(scene.DIAMONDS_SEED + 2)  # build_diamonds_scene's palette picks, in order
        got = [(np.array(c, dtype=float), int(rng.integers(1 << 30))) for c, _ in diamonds]
        if len(self._layouts) >= 64:
            self._layouts.clear()
        self._layouts[keys] = got
        return got


def norm_key(k):
    """A tile key as a hashable (x, y) tuple, rounded as diamonds_map.tile_key rounds."""
    return round(float(k[0]), 1), round(float(k[1]), 1)


def chunk_rects(size, n=CHUNKS, grids=tuple(GRIDS)):
    """[{"id", "grid", "row", "col", "rect": (x0, y0, w, h)}] of every chunk of the given grids ("grid", "shift_x",
    "shift_y", "shift_xy"): an n x n grid of the world, shifted by half a chunk for the shifted ones (their last
    row / column reaches past the world's right / bottom edge: it wraps)."""
    W, H = size
    cw, ch = W / n, H / n
    out = []
    for g in grids:
        fx, fy = GRIDS[g]
        for r in range(n):
            for c in range(n):
                out.append({"id": f"{g} r{r}c{c}", "grid": g, "row": r, "col": c,
                            "rect": ((c + fx) * cw, (r + fy) * ch, cw, ch)})
    return out


def find_chunk(size, chunk_id):
    for ch in chunk_rects(size):
        if ch["id"] == chunk_id:
            return ch
    raise KeyError(chunk_id)


def _clip_area(poly, x0, y0, x1, y1):
    """Area of the convex polygon (2, n) clipped to the box (Sutherland-Hodgman)."""
    pts = list(map(tuple, np.asarray(poly).T))
    for axis, bound, keep_lo in ((0, x0, False), (0, x1, True), (1, y0, False), (1, y1, True)):
        if not pts:
            return 0.0
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            ina = a[axis] <= bound if keep_lo else a[axis] >= bound
            inb = b[axis] <= bound if keep_lo else b[axis] >= bound
            if ina != inb:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if inb:
                out.append(b)
        pts = out
    if len(pts) < 3:
        return 0.0
    x, y = np.array(pts).T
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def chunk_tiles(info, rect, keys=None):
    """[(polygon (2, n) in map coordinates, possibly shifted by a world size across the wrap, palette pick)] of the
    tiles overlapping the chunk rect, and the area of the rect they cover. keys: a layout (tile keys of the map's
    tiling) instead of the map's own tiles."""
    W, H = info.size
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    out, covered = [], 0.0
    tiles = zip(info.tiles, info.tile_picks) if keys is None else info.layout_tiles(keys)
    for poly, pick in tiles:
        for sx in (-W, 0.0, W):
            for sy in (-H, 0.0, H):
                p = poly + np.array([[sx], [sy]])
                if p[0].max() <= x0 or p[0].min() >= x1 or p[1].max() <= y0 or p[1].min() >= y1:
                    continue
                a = _clip_area(p, x0, y0, x1, y1)
                if a > 0:
                    out.append((p, pick))
                    covered += a
    return out, covered


def _tile_field(tiles, rect, reach=40.0):
    """(rows, cols) grid at 1 world unit spacing over the chunk rect (sample (i, j) at (x0 + j, y0 + i)): each
    sample's distance to the nearest tile polygon (0 inside one), capped at `reach` (farther than any shape needs).
    A point's true distance is at least its nearest sample's minus FIELD_SLACK."""
    x0, y0, w, h = rect
    cols, rows = int(math.ceil(w)) + 1, int(math.ceil(h)) + 1
    field = np.full((rows, cols), reach)
    for poly in tiles:
        j0 = max(0, int(math.floor(poly[0].min() - reach - x0)))
        j1 = min(cols, int(math.ceil(poly[0].max() + reach - x0)) + 1)
        i0 = max(0, int(math.floor(poly[1].min() - reach - y0)))
        i1 = min(rows, int(math.ceil(poly[1].max() + reach - y0)) + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        px, py = np.meshgrid(x0 + np.arange(j0, j1), y0 + np.arange(i0, i1))
        a, b = poly, np.roll(poly, -1, axis=1)
        d = np.full(px.shape, np.inf)
        inside = np.zeros(px.shape, dtype=bool)
        for k in range(poly.shape[1]):
            ax, ay, bx, by = a[0, k], a[1, k], b[0, k], b[1, k]
            ex, ey = bx - ax, by - ay
            t = np.clip(((px - ax) * ex + (py - ay) * ey) / max(ex * ex + ey * ey, 1e-12), 0, 1)
            d = np.minimum(d, np.hypot(px - ax - t * ex, py - ay - t * ey))
            straddle = (ay > py) != (by > py)
            xc = ax + (py - ay) * ex / np.where(straddle, ey, 1.0)
            inside ^= straddle & (px < xc)
        d[inside] = 0.0
        field[i0:i1, j0:j1] = np.minimum(field[i0:i1, j0:j1], d)
    return field


class SpawnDraw:
    """Shape specs {"kind", "r" (placement circle radius), "size" (pumpkin half width), "sides"} in the map's spawn
    order: the k-th is a pumpkin when floor((k + 1) x pumpkin_share) > floor(k x share) (scene.spawn_kinds), the
    others take the side counts in turn; sizes uniform in their ranges (as scene.build_diamonds_scene), or with
    `extremes` (EXTREMES) only in that share of them at the stuck-prone end: the top for pumpkins and polygons of 7+
    sides, the bottom for fewer sides."""

    def __init__(self, spawn, rng, extremes=None):
        import pumpkins
        self.spawn, self.rng = spawn, rng
        self.share = float(spawn.get("pumpkin_share", 0.0))
        self.r_range = tuple(float(v) for v in spawn["sizes"])
        self.p_range = tuple(float(v) for v in spawn.get("pumpkin_sizes", pumpkins.SIZES))
        self.reach = pumpkins.REACH
        self.extremes = extremes
        self.k = self.others = 0

    def _size(self, lo, hi, top):
        if self.extremes is not None:
            span = (hi - lo) * float(self.extremes)
            lo, hi = (hi - span, hi) if top else (lo, lo + span)
        return float(self.rng.uniform(lo, hi))

    def next(self):
        k, sides = self.k, self.spawn["sides"]
        self.k += 1
        if math.floor((k + 1) * self.share) > math.floor(k * self.share):
            size = self._size(*self.p_range, True)
            return {"kind": "pumpkin", "size": size, "r": size * self.reach}
        self.others += 1
        n = int(sides[(self.others - 1) % len(sides)])
        return {"kind": "polygon", "r": self._size(*self.r_range, n >= 7), "sides": n}

    def take(self, area):
        """The next specs, until their circles' area reaches `area`."""
        out, got = [], 0.0
        while got < area:
            out.append(self.next())
            got += math.pi * out[-1]["r"] ** 2
        return out


def _polygon_area(points):
    x, y = points
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


class ChunkTest:
    """One chunk test's world, ready to step (see the module doc). Local coordinates: the world spans
    (0, 0) .. world_size and the chunk rect sits at `rect` (offset by the margin); map = local + `offset`.
    keys: test this layout (tile keys of the map's tiling) instead of the map's own tiles.
    sample_dt: also record every shape's centroid (mean of its points, map coordinates, not wrapped) every sample_dt
    seconds (a whole number of frames), from t = 0 until it clears; result() then holds "track" (see track()).
    Read only: the simulation is bit-identical with and without it."""

    def __init__(self, info, chunk, seed, fill=FILL, seconds=SECONDS, vel=VEL, margin=MARGIN, keys=None,
                 sample_dt=None, extremes=EXTREMES):
        from entities import EnvConfig, FixedBody, SoftBody, polygon_points
        import pumpkins
        from solver import World
        EnvConfig.backend = "cpu"  # bit-deterministic (the GPU path uses float32 atomics)
        self.info, self.chunk, self.seed = info, chunk, seed
        self.fill, self.seconds, self.vel = fill, seconds, vel
        x0, y0, w, h = chunk["rect"]
        mx, my = margin * w, margin * h
        self.world_size = (w + 2 * mx, h + 2 * my)
        self.rect = (mx, my, w, h)  # local
        self.offset = np.array([x0 - mx, y0 - my])
        tiles, covered = chunk_tiles(info, chunk["rect"], keys)
        self.free_area = w * h - covered
        world = World()
        world.show_grid = False
        bodies = []
        local_tiles = []
        for poly, pick in tiles:
            pts = poly - self.offset[:, None]
            local_tiles.append(pts)
            b = FixedBody(pts, point_radius=2)
            b.palette_role, b.palette_pick = "diamond", pick
            bodies.append(b)
        self.tile_bodies = list(bodies)
        if not bodies:  # World.remove needs one body left: a tiny fixed segment far off, touching nothing
            b = FixedBody(np.array([[-1e4, -1e4 + 1], [-1e4, -1e4]]), point_radius=0)
            b.palette_role = None
            bodies.append(b)
        # --- shapes: fill x the free area of placement circles (see _place), then their bodies
        rng = np.random.default_rng([int(seed), 7177])
        target = fill * self.free_area
        placed, placed_area = self._place(SpawnDraw(info.spawn, rng, extremes), local_tiles, rng, target)
        self.fill_achieved = placed_area / max(self.free_area, 1e-9)
        self.shapes = []
        pick_rng = np.random.default_rng([int(seed), 7178])
        for spec, (cx, cy, angle) in placed:
            if vel is None:
                v0 = (float(rng.normal(VEL_MEAN[0], VEL_STD[0])), float(rng.normal(VEL_MEAN[1], VEL_STD[1])))
            else:
                speed, d = float(rng.uniform(0, vel)), float(rng.uniform(0, 2 * math.pi))
                v0 = (speed * math.cos(d), speed * math.sin(d))
            if spec["kind"] == "pumpkin":  # as scene.build_diamonds_scene: upright, leaning up to 30 degrees
                b = pumpkins.make(cx, cy, spec["size"], tilt=angle - math.pi / 6, point_radius=2)
            else:
                b = SoftBody(polygon_points(cx, cy, spec["r"], spec["sides"], angle=-math.pi / 2 + angle),
                             stiffness=0.1, skip_sets=2, point_radius=2)
            b.vels[:] = np.array([[v0[0]], [v0[1]]])
            b.palette_role, b.palette_pick = "hexagon", int(pick_rng.integers(1 << 30))
            b.chunk_area = _polygon_area(b.points)
            b.chunk_r, b.chunk_c = spec["r"], (cx, cy)  # its placement circle
            self.shapes.append(b)
        bodies += self.shapes
        world.add_many(bodies)
        self.world = world
        self.points = int(world.P.shape[1])
        self.frame = 0
        self.frames = int(round(seconds * FPS))
        self.cleared = []  # (body, time s)
        self.remaining = list(self.shapes)
        self._history = []  # per frame: centres of the remaining shapes (for the SLOW_WINDOW speeds)
        self.sample_dt = sample_dt
        self._track = None
        if sample_dt is not None:
            every = sample_dt * FPS
            if every < 1 or abs(every - round(every)) > 1e-9:
                raise ValueError(f"sample_dt {sample_dt} isn't a whole number of 1/{FPS} s frames")
            self._sample_every = int(round(every))
            self._track = [[] for _ in self.shapes]
            self._sample()

    def _sample(self):
        """Record the remaining shapes' centroids (map coordinates) in the track (sample_dt)."""
        live = {id(b) for b in self.remaining}
        for b, xy in zip(self.shapes, self._track):
            if id(b) in live:
                c = b.points.mean(axis=1) + self.offset
                xy.append([round(float(c[0]), 2), round(float(c[1]), 2)])

    def track(self):
        """The sampled centroids (sample_dt set; else None): {"sample_dt", "shapes": [{"r" (placement circle
        radius), "area" (polygon area), "exit_t" (clear time s, None if still in), "xy": [[x, y], ...] at
        t = k x sample_dt, k = 0, 1, ... while the shape is in the chunk}]} in shape order."""
        if self._track is None:
            return None
        exit_t = {id(b): t for b, t in self.cleared}
        return {"sample_dt": self.sample_dt,
                "shapes": [{"r": round(b.chunk_r, 3), "area": round(b.chunk_area, 2),
                            "exit_t": None if id(b) not in exit_t else round(exit_t[id(b)], 4), "xy": xy}
                           for b, xy in zip(self.shapes, self._track)]}

    def _place(self, draw, tiles, rng, target):
        """[(spec, (cx, cy, angle))] and their circles' area. Each circle lies wholly inside the chunk rect, GAP clear
        of the tiles and the others. First the shapes drawn in spawn order (SpawnDraw) whose circles add up to
        `target`, placed largest first (ones that don't fit are skipped); then, while the target isn't covered, more
        shapes drawn one at a time in spawn order, each placed if it fits, until MAX_MISSES in a row didn't (near
        saturation only the smaller ones fit, so this top-up leans small)."""
        x0, y0, w, h = self.rect
        field = _tile_field(tiles, self.rect)
        r_low = float(draw.spawn["sizes"][0])  # the smallest shapes: the most that could fit
        circles = np.empty((int(w * h / (math.pi * r_low * r_low)) + 16, 3))
        placed, area = [], 0.0
        for spec in sorted(draw.take(target), key=lambda s: -s["r"]):
            if area >= target:
                break
            if self._place_one(spec, field, circles, placed, rng):
                area += math.pi * spec["r"] ** 2
        misses = 0
        while area < target and misses < MAX_MISSES:
            spec = draw.next()
            if self._place_one(spec, field, circles, placed, rng):
                area += math.pi * spec["r"] ** 2
                misses = 0
            else:
                misses += 1
        return placed, area

    def _place_one(self, spec, field, circles, placed, rng):
        """Place one shape's circle (see _place) if a spot is found in TRIES random positions (True if placed)."""
        x0, y0, w, h = self.rect
        r = spec["r"]
        if 2 * (r + GAP) > min(w, h):
            return False
        n = len(placed)
        for _ in range(TRIES // CANDIDATES):
            cx = rng.uniform(x0 + r + GAP, x0 + w - r - GAP, CANDIDATES)
            cy = rng.uniform(y0 + r + GAP, y0 + h - r - GAP, CANDIDATES)
            ix = np.clip(np.rint(cx - x0).astype(int), 0, field.shape[1] - 1)
            iy = np.clip(np.rint(cy - y0).astype(int), 0, field.shape[0] - 1)
            clear = field[iy, ix] - FIELD_SLACK - r  # room left to the nearest tile (a lower bound)
            if n:
                d = np.hypot(cx[:, None] - circles[None, :n, 0], cy[:, None] - circles[None, :n, 1])
                clear = np.minimum(clear, (d - circles[None, :n, 2]).min(axis=1) - r)
            ok = np.flatnonzero(clear >= GAP)
            if len(ok):
                ok = ok[:SNUG]  # the snuggest of the first SNUG valid candidates (1: plain random placement)
                k = ok[int(np.argmin(clear[ok]))]
                circles[n] = cx[k], cy[k], r
                placed.append((spec, (float(cx[k]), float(cy[k]), float(rng.uniform(0, math.pi / 3)))))
                return True
        return False

    @property
    def time(self):
        return self.frame / FPS

    @property
    def done(self):
        return self.frame >= self.frames

    def step(self):
        """One frame: World.step(1/60) (4 ticks, one CCD span), then remove the shapes that left the chunk rect."""
        self.world.step(1.0 / FPS)
        self.frame += 1
        x0, y0, w, h = self.rect
        gone = []
        for b in self.remaining:
            p = b.points
            if p[1].min() >= y0 + h or p[1].max() <= y0 or p[0].min() >= x0 + w or p[0].max() <= x0:
                gone.append(b)
        if gone:
            ids = {id(b) for b in gone}
            self.remaining = [b for b in self.remaining if id(b) not in ids]
            self.cleared += [(b, self.time) for b in gone]
            self.world.remove(bodies=gone)
        self._history.append({id(b): b.points.mean(axis=1).copy() for b in self.remaining})
        if len(self._history) > int(round(SLOW_WINDOW * FPS)) + 1:
            self._history.pop(0)
        if self._track is not None and self.frame % self._sample_every == 0:
            self._sample()

    def speeds(self):
        """{id(body): mean speed over the last SLOW_WINDOW s (or what has run of it), px/s} of the remaining shapes."""
        out = {}
        if len(self._history) < 2:
            return {id(b): 0.0 for b in self.remaining}
        first, last = self._history[0], self._history[-1]
        dt = (len(self._history) - 1) / FPS
        for b in self.remaining:
            k = id(b)
            out[k] = float(np.hypot(*(last[k] - first[k])) / dt) if k in first and k in last else 0.0
        return out

    def score(self):
        n = len(self.shapes)
        slow = sum(v < SLOW_SPEED for v in self.speeds().values())
        area = sum(b.chunk_area for b in self.shapes)
        T = self.seconds
        return {
            "score": len(self.cleared) / n if n else 1.0,
            "area_score": sum(b.chunk_area for b, _ in self.cleared) / area if area else 1.0,
            "flow": sum(1.0 - t / T for _, t in self.cleared) / n if n else 1.0,
            "free": 1.0 - slow / n if n else 1.0,
        }

    def result(self):
        W, H = self.info.size
        sp = self.speeds()
        stuck = []
        for b in self.remaining:
            c = b.points.mean(axis=1) + self.offset
            stuck.append([round(float(c[0] % W), 1), round(float(c[1] % H), 1), round(sp[id(b)], 1),
                          round(b.chunk_r, 2)])
        return {"chunk": self.chunk["id"], "rect": [round(v, 2) for v in self.chunk["rect"]], "seed": self.seed,
                **{k: round(v, 4) for k, v in self.score().items()},
                "shapes": len(self.shapes), "cleared": len(self.cleared), "stuck_count": len(self.remaining),
                "slow_count": sum(v < SLOW_SPEED for v in sp.values()),
                "tiles": len(self.tile_bodies), "fill": self.fill, "fill_achieved": round(self.fill_achieved, 4),
                "points": self.points, "sim_seconds": self.time,
                "stuck": stuck,
                **({"track": self.track()} if self._track is not None else {})}


def run_chunk(info, chunk, seed, **kw):
    """Build and run one chunk test to the end; its result dict plus "wall" (s, build + run) and "build_wall".
    kw: ChunkTest's settings (fill, seconds, vel, margin, keys, sample_dt, extremes)."""
    t0 = time.perf_counter()
    test = ChunkTest(info, chunk, seed, **kw)
    t1 = time.perf_counter()
    while not test.done:
        test.step()
    res = test.result()
    res["build_wall"] = round(t1 - t0, 3)
    res["wall"] = round(time.perf_counter() - t0, 3)
    return res


def run_chunk_keys(info, keys, chunk_id, seed, **kw):
    """run_chunk on a layout: chunk `chunk_id` (an id like "grid r1c2", or a chunk_rects entry) of the world built
    from `keys` (tile keys of the map's tiling) instead of the map's own tiles. The same result dict."""
    chunk = find_chunk(info.size, chunk_id) if isinstance(chunk_id, str) else chunk_id
    return run_chunk(info, chunk, seed, keys=keys, **kw)


_WORKER = {}


def _watch_parent():
    """A pool worker's watchdog thread: exit when the parent process is gone (the worker got re-parented), so a
    killed run leaves no orphan workers."""
    import threading
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def _init_worker(name, kw=None):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _watch_parent()
    from entities import EnvConfig
    EnvConfig.backend = "cpu"
    _WORKER["info"] = MapInfo(name)
    _WORKER["kw"] = dict(kw or {})


def _job(args):
    chunk, seed, kw = args
    return run_chunk(_WORKER["info"], chunk, seed, **kw)


def _pool_job(job):
    keys, chunk_id, seed = job[:3]
    kw = dict(_WORKER["kw"], **(job[3] if len(job) > 3 else {}))
    return run_chunk_keys(_WORKER["info"], keys, chunk_id, seed, **kw)


def _pool_whole(job):
    keys, seconds, seed = job
    return whole_map_score(_WORKER["info"].path, keys, seconds, seed)


class ChunkPool:
    """A persistent pool of `workers` processes (spawn start method; each loads the map and compiles numba once) for
    scoring many chunk tests of many layouts. Use it from a script file (spawn re-imports __main__), as a context
    manager:

        with ChunkPool("evolved 1 halloween", 6) as pool:
            results = pool.score([(keys, "grid r1c2", seed), ...])

    score(jobs): jobs = [(keys, chunk_id, seed)] (an optional 4th item: a dict of ChunkTest settings for that job
    over the pool's own `kw`: fill, seconds, vel, sample_dt) -> their run_chunk_keys results, in job order;
    submit(job) queues one without waiting. Each job's keys go
    with it (pickled, ~20 KB: nothing next to a test); each worker caches its last 64 layouts' tiles.
    whole(jobs): jobs = [(keys, seconds, seed)] -> whole_map_score results, in order, on the same workers.
    Leaving the block (or close()) shuts the workers down and waits for them; an exception terminates them. Workers
    also exit by themselves within a second if the parent process dies."""

    def __init__(self, map_name, workers, **kw):
        import multiprocessing as mp
        self.map_path = map_path(map_name)
        self.workers = int(workers)
        self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_worker,
                                                  initargs=(self.map_path, kw))

    def score(self, jobs):
        jobs = [(frozenset(norm_key(k) for k in j[0]),) + tuple(j[1:]) for j in jobs]
        return self._pool.map(_pool_job, jobs, chunksize=1)

    def submit(self, job, callback=None, error_callback=None):
        """Queue one score() job without waiting: its multiprocessing AsyncResult (callback(result), or
        error_callback(exception), runs in the pool's result thread when it is done)."""
        job = (frozenset(norm_key(k) for k in job[0]),) + tuple(job[1:])
        return self._pool.apply_async(_pool_job, (job,), callback=callback, error_callback=error_callback)

    def whole(self, jobs):
        jobs = [(sorted(norm_key(k) for k in keys), seconds, seed) for keys, seconds, seed in jobs]
        return self._pool.map(_pool_whole, jobs, chunksize=1)

    def pids(self):
        """The worker processes' pids (for tests)."""
        return [p.pid for p in self._pool._pool]

    def close(self, terminate=False):
        if self._pool is None:
            return
        if terminate:
            self._pool.terminate()
        else:
            self._pool.close()
        self._pool.join()
        self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(terminate=exc_type is not None)


def whole_map_score(map_name, keys, seconds, seed, weights=(0.7, 0.2, 0.1)):
    """One whole-map run of the layout `keys` (tile keys of the map's tiling) scored exactly as evolve_layout scores
    its runs (evolve_layout.run: `seconds` of sim time in the map's spawn settings and tiling, shape spawn seed
    `seed`, score = w1 spread + w2 unstuck + w3 tile evenness over the last 10 s): run's result dict (score, spread,
    unstuck, slow_speed, even, diamonds, keys, cells, stuck_xy, secs, ...). In this process; ChunkPool.whole runs
    several in parallel."""
    import scene
    import evolve_layout
    path = map_path(map_name)
    with open(path) as f:
        size = tuple(float(v) for v in json.load(f).get("size", (1800.0, 1169.0)))
    cwd = os.getcwd()
    try:
        evolve_layout.init(size, scene.diamonds_spawn(path), tuple(weights), scene._layout_tiling(path))
        return evolve_layout.run({"tag": "whole", "keys": sorted(norm_key(k) for k in keys),
                                  "frames": int(round(seconds * 60)), "seed": seed})
    finally:
        os.chdir(cwd)  # (init moves to the repo)


def main(argv=None):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless (the viewer imports this module with a display)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("map", help='a map name in maps/penrose (e.g. "evolved 1 halloween") or a layout file')
    p.add_argument("--seeds", type=int, default=3, help="seeds 1..N per chunk")
    p.add_argument("--grids", default="grid", help='comma list of grid, shift_x, shift_y, shift_xy, or "all"')
    p.add_argument("--chunks", default=None, help='comma list of chunk ids to run (e.g. "grid r1c2,shift_x r0c3"; any grid), '
                                                    'default all of --grids')
    p.add_argument("--fill", type=float, default=FILL)
    p.add_argument("--seconds", type=float, default=SECONDS)
    p.add_argument("--vel", type=float, default=VEL, help="speed 0..VEL any way (default: like the whole map)")
    p.add_argument("--workers", type=int, default=1, help="processes (1: in this process)")
    p.add_argument("--out", default=None, help="JSON output path (default: print only)")
    args = p.parse_args(argv)
    grids = tuple(GRIDS) if args.grids == "all" else tuple(g.strip() for g in args.grids.split(","))
    t0 = time.perf_counter()
    info = MapInfo(args.map)
    load = time.perf_counter() - t0
    chunks = chunk_rects(info.size, grids=grids)
    if args.chunks:  # by id, from any grid (overrides --grids)
        want = [c.strip() for c in args.chunks.split(",")]
        chunks = [find_chunk(info.size, c) for c in want]
    kw = {"fill": args.fill, "seconds": args.seconds, "vel": args.vel}
    jobs = [(c, s, kw) for c in chunks for s in range(1, args.seeds + 1)]
    print(f"{info.name}: {len(chunks)} chunks x {args.seeds} seeds, fill {args.fill:g}, {args.seconds:g} s, "
          f"map load {load:.1f} s", flush=True)
    t1 = time.perf_counter()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(info.path,)) as pool:
            results = pool.map(_job, jobs, chunksize=1)
    else:
        from entities import EnvConfig
        EnvConfig.backend = "cpu"
        run_chunk(info, chunks[0], 0, **dict(kw, seconds=0.1))  # warm up numba (compile) outside the timings
        results = [run_chunk(info, c, s, **kw) for c, s, kw in jobs]
    total = time.perf_counter() - t1
    by_chunk = {}
    for r in results:
        by_chunk.setdefault(r["chunk"], []).append(r)
    print(f"{'chunk':16s} {'score':>6s} {'std':>5s} {'area':>6s} {'flow':>6s} {'shapes':>6s} {'fill':>5s} "
          f"{'tiles':>5s} {'wall s':>6s}  per seed")
    summary = []
    for c in chunks:
        rs = by_chunk[c["id"]]
        sc = np.array([r["score"] for r in rs])
        row = {"chunk": c["id"], "rect": rs[0]["rect"], "score": float(sc.mean()), "score_std": float(sc.std()),
               "area_score": float(np.mean([r["area_score"] for r in rs])),
               "flow": float(np.mean([r["flow"] for r in rs])),
               "free": float(np.mean([r["free"] for r in rs])),
               "wall": float(np.mean([r["wall"] for r in rs]))}
        summary.append(row)
        print(f"{c['id']:16s} {row['score']:6.3f} {row['score_std']:5.3f} {row['area_score']:6.3f} {row['flow']:6.3f} "
              f"{np.mean([r['shapes'] for r in rs]):6.0f} {np.mean([r['fill_achieved'] for r in rs]):5.2f} "
              f"{rs[0]['tiles']:5d} {row['wall']:6.2f}  " + " ".join(f"{v:.3f}" for v in sc), flush=True)
    noise = {}
    for key in ("score", "area_score", "flow", "free"):
        per = [np.array([r[key] for r in by_chunk[c["id"]]]) for c in chunks]
        means = np.array([v.mean() for v in per])
        within = float(np.sqrt(np.mean([v.var() for v in per])))  # rms of the per-chunk std across seeds
        noise[key] = {"between_std": float(means.std()), "within_rms_std": within,
                      "min": float(means.min()), "max": float(means.max())}
        print(f"{key:10s}: between chunks std {means.std():.3f} (chunk means {means.min():.3f}-{means.max():.3f}), "
              + (f"within chunk (seed noise) rms std {within:.3f}, ratio {means.std() / max(within, 1e-9):.1f}"
                 if args.seeds > 1 else "(one seed: no seed noise)"))
    walls = np.array([r["wall"] for r in results])
    print(f"wall per test {walls.mean():.2f} s (min {walls.min():.2f}, max {walls.max():.2f}; build "
          f"{np.mean([r['build_wall'] for r in results]):.2f} s of it); total {total:.1f} s with "
          f"{args.workers} worker(s)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"map": info.name, "settings": dict(kw, seeds=args.seeds, grids=list(grids)),
                       "summary": summary, "noise": noise, "runs": results}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
