"""Per-chunk stuckness test of a penrose / diamonds map: how well does one region of the map drain a crowd of shapes?

    python3 chunk_test.py "evolved 1 halloween" [--seeds 3] [--grids grid] [--fill 0.6] [--seconds 5]
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
    `fill` (default 0.6) of the chunk's free area (rect area minus the tiles' area inside it). Shapes for the whole
    target are drawn first and placed largest first, each at the snuggest of a few random free spots (SNUG); ones
    that don't fit are skipped, then more are drawn one at a time to top up (see ChunkTest._place). Random placement
    saturates: on evolved 1 halloween 0.5 is always met, 0.6 gives ~0.56-0.58 and 0.7 only ~0.59-0.60, so the
    achieved fill is reported ("fill_achieved"). Each shape gets a random velocity of uniform direction and speed
    0..`vel` (default 30 px/s).
  * Run `seconds` (default 5) of sim time. Once per frame, a shape whose bounding box lies wholly outside the chunk
    rect (below it, or off a side) has cleared the chunk: its clear time is recorded and it is removed, so nothing
    piles up and it can't come back.
  * Score:
        score       = cleared shapes / shapes                 (1 = the chunk drained completely, 0 = nothing left)
        area_score  = cleared shape area / shape area          (the same weighted by polygon area)
        flow        = mean over shapes of 1 - t_clear / T      (a shape still there at T counts 0; rewards fast
                                                               draining, so it separates chunks that all clear)
    and the shapes still in the chunk at the end are "stuck": their centres (map coordinates, wrapped) and their
    speed over the last second go into "stuck" ([x, y, px/s, radius]) for a heatmap.

Deterministic for a given (map, chunk, seed, settings): the seed draws the shapes, their placement and velocities.
The CLI runs every chunk of the chosen grids on N seeds (seed = 1..N), prints a per-chunk table (mean, std across
seeds, wall time) and writes JSON. chunk_viewer.py shows the same test in real time.
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
FILL = 0.6  # share of the chunk's free area covered by the shapes' placement circles
SECONDS = 5.0  # sim time per test
VEL = 30.0  # px/s: each shape's initial speed is uniform in 0..VEL, direction uniform
SLOW_SPEED = 10.0  # px/s: a remaining shape slower than this over the last second is "slow" (resting, not draining)
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
        from diamonds_map import layout_lighting
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


def chunk_tiles(info, rect):
    """[(polygon (2, n) in map coordinates, possibly shifted by a world size across the wrap, palette pick)] of the
    tiles overlapping the chunk rect, and the area of the rect they cover."""
    W, H = info.size
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    out, covered = [], 0.0
    for poly, pick in zip(info.tiles, info.tile_picks):
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
    others take the side counts in turn; sizes uniform in their ranges (as scene.build_diamonds_scene)."""

    def __init__(self, spawn, rng):
        import pumpkins
        self.spawn, self.rng = spawn, rng
        self.share = float(spawn.get("pumpkin_share", 0.0))
        self.r_range = tuple(float(v) for v in spawn["sizes"])
        self.p_range = tuple(float(v) for v in spawn.get("pumpkin_sizes", pumpkins.SIZES))
        self.reach = pumpkins.REACH
        self.k = self.others = 0

    def next(self):
        k, sides = self.k, self.spawn["sides"]
        self.k += 1
        if math.floor((k + 1) * self.share) > math.floor(k * self.share):
            size = float(self.rng.uniform(*self.p_range))
            return {"kind": "pumpkin", "size": size, "r": size * self.reach}
        self.others += 1
        return {"kind": "polygon", "r": float(self.rng.uniform(*self.r_range)),
                "sides": int(sides[(self.others - 1) % len(sides)])}

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
    (0, 0) .. world_size and the chunk rect sits at `rect` (offset by the margin); map = local + `offset`."""

    def __init__(self, info, chunk, seed, fill=FILL, seconds=SECONDS, vel=VEL, margin=MARGIN):
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
        tiles, covered = chunk_tiles(info, chunk["rect"])
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
        placed, placed_area = self._place(SpawnDraw(info.spawn, rng), local_tiles, rng, target)
        self.fill_achieved = placed_area / max(self.free_area, 1e-9)
        self.shapes = []
        pick_rng = np.random.default_rng([int(seed), 7178])
        for spec, (cx, cy, angle) in placed:
            speed, d = float(rng.uniform(0, vel)), float(rng.uniform(0, 2 * math.pi))
            if spec["kind"] == "pumpkin":  # as scene.build_diamonds_scene: upright, leaning up to 30 degrees
                b = pumpkins.make(cx, cy, spec["size"], tilt=angle - math.pi / 6, point_radius=2)
            else:
                b = SoftBody(polygon_points(cx, cy, spec["r"], spec["sides"], angle=-math.pi / 2 + angle),
                             stiffness=0.1, skip_sets=2, point_radius=2)
            b.vels[:] = np.array([[speed * math.cos(d)], [speed * math.sin(d)]])
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
        self._history = []  # per frame: centres of the remaining shapes (for the last-second speeds)

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
        if len(self._history) > FPS + 1:
            self._history.pop(0)

    def speeds(self):
        """{id(body): mean speed over the last second (or what has run of it), px/s} of the remaining shapes."""
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
        area = sum(b.chunk_area for b in self.shapes)
        T = self.seconds
        return {
            "score": len(self.cleared) / n if n else 1.0,
            "area_score": sum(b.chunk_area for b, _ in self.cleared) / area if area else 1.0,
            "flow": sum(1.0 - t / T for _, t in self.cleared) / n if n else 1.0,
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
                "stuck": stuck}


def run_chunk(info, chunk, seed, **kw):
    """Build and run one chunk test to the end; its result dict plus "wall" (s, build + run) and "build_wall"."""
    t0 = time.perf_counter()
    test = ChunkTest(info, chunk, seed, **kw)
    t1 = time.perf_counter()
    while not test.done:
        test.step()
    res = test.result()
    res["build_wall"] = round(t1 - t0, 3)
    res["wall"] = round(time.perf_counter() - t0, 3)
    return res


_WORKER = {}


def _init_worker(name):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from entities import EnvConfig
    EnvConfig.backend = "cpu"
    _WORKER["info"] = MapInfo(name)


def _job(args):
    chunk, seed, kw = args
    return run_chunk(_WORKER["info"], chunk, seed, **kw)


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
    p.add_argument("--vel", type=float, default=VEL)
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
               "wall": float(np.mean([r["wall"] for r in rs]))}
        summary.append(row)
        print(f"{c['id']:16s} {row['score']:6.3f} {row['score_std']:5.3f} {row['area_score']:6.3f} {row['flow']:6.3f} "
              f"{np.mean([r['shapes'] for r in rs]):6.0f} {np.mean([r['fill_achieved'] for r in rs]):5.2f} "
              f"{rs[0]['tiles']:5d} {row['wall']:6.2f}  " + " ".join(f"{v:.3f}" for v in sc), flush=True)
    noise = {}
    for key in ("score", "area_score", "flow"):
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
