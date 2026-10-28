"""gl_world.WorldPacker's caches (the per-body attribute arrays and the still bodies' rows, _Attrs / _Still) and the
material table's style_version key must give exactly what packing from scratch gives. Run from the repo root:

    .venv/bin/python3 tests/pack_cache_check.py [scene ...]

Part 1 (numpy only): long-lived packers (one unlit, one lit) against a fresh WorldPacker every call, over moving frames of several
maps, cycling the modes (points / filled / no outlines, lit material ids and unlit, the lit points view's two packs
a frame) and along the way: a grab, a palette switch, a material id change, a colour change on one still tile, a
still tile moved in place (an editor move: no repack) and moved back, a body added and one removed (repacks), a zoom
change. Every array of every pack must be equal (dtype and values).
Part 2 (GL): GLWorld._materials redoes the table when a fill colour, material or material_over is assigned (and
_mat_sig is then a new object), and not otherwise; its ids match a fresh GLWorld's.
Exit status 1 on any mismatch.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from entities import EnvConfig, FixedBody, SoftBody, polygon_points  # noqa: E402
from gl_world import WorldPacker  # noqa: E402
import palettes  # noqa: E402
from scene import PRESETS  # noqa: E402

WORLD = (1800, 1169)
SCENES = ["penrose/evolved 1 halloween", "penrose/evolved 1", "diamonds/evolved 1", "fluid", "newtons cradle"]
MODES = [(False, True), (True, True), (True, False)]
bad = 0


def same(a, b, what):
    global bad
    for k in b:
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
            bad += 1
            print(f"MISMATCH {what}: {k} {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
            return False
    return True


def still_tile(world):
    return next((b for b in world.bodies if isinstance(b, FixedBody) and b.points.shape[1] >= 3
                 and getattr(b, "wall_side", None) is None), None)


def run(name, frames=48):
    world = PRESETS[name](*WORLD)
    packers = {False: WorldPacker(), True: WorldPacker()}  # unlit / lit: one GLWorld renders one way at a time
    zoom = 0.5
    mats = mats_pts = None
    n = 0
    for f in range(frames):
        world.step(1 / 60)
        nb = len(world.bodies)
        if mats is None or len(mats) != nb:
            mats = (np.arange(nb) % 5 + 1).astype(np.int64)
            mats_pts = mats + 10  # (the points view's twin ids: the same array until the materials change)
        tile = still_tile(world)
        if f == 6:
            movable = np.flatnonzero(world.W > 0)
            if len(movable):
                world.grab(world.P[:, movable[len(movable) // 2]].copy(), 60.0)
        if f == 10 and getattr(world, "palette", None) is not None:
            names = [p for p in palettes.PALETTES if p != world.palette]
            palettes.apply_palette(world, names[0])
        if f == 14:
            mats = mats.copy()
            mats[::3] = 2  # a new material id array (the lit renderer's body_mat after a material change)
            mats_pts = mats + 10
        if f == 18 and tile is not None:
            tile.fill_col = [250, 10, 10]
        if f == 22 and tile is not None:
            tile.points += 7.0  # an editor move: in place, no repack
        if f == 26 and tile is not None:
            tile.points -= 7.0
        if f == 30:
            world.add(SoftBody(polygon_points(WORLD[0] / 2, 80, 20, 6), stiffness=0.5, point_radius=2))
            mats = None
            continue
        if f == 36:
            world.remove([world.bodies[-1]])
            mats = None
            continue
        if f == 40:
            zoom = 0.75
        for filled, outlines in MODES:
            for lit in (False, True):
                if lit and not filled:  # the lit points view: its points-view pack, then the occluder fill's
                    calls = [(False, mats_pts), (True, mats)]
                else:
                    calls = [(filled, mats if lit else None)]
                for fl, bm in calls:
                    a = packers[lit].pack(world, fl, zoom, outlines, bm, 1, 7)
                    b = WorldPacker().pack(world, fl, zoom, outlines, bm, 1, 7)
                    same(a, b, f"{name} frame {f} filled={fl} outlines={outlines} lit={lit}")
                    n += 1
    print(f"{name}: {n} packs compared", flush=True)


def materials_check():
    """Part 2: GLWorld._materials keyed on entities.style_version."""
    global bad
    import moderngl
    import pygame
    from gl_world import GLWorld
    pygame.init()
    ctx = moderngl.create_standalone_context(require=330)
    world = PRESETS["penrose/evolved 1 halloween"](*WORLD)
    gw = GLWorld(ctx, (64, 64))

    def check(label, expect_new):
        global bad
        sig = gw._mat_sig
        gw._materials(world)
        fresh = GLWorld(ctx, (64, 64))
        fresh._materials(world)
        changed = gw._mat_sig is not sig
        ok = changed == expect_new and np.array_equal(gw.body_mat, fresh.body_mat) and \
            gw.n_materials == fresh.n_materials and gw._mat_bytes == fresh._mat_bytes
        if not ok:
            bad += 1
        print(f"{'ok  ' if ok else 'FAIL'} materials: {label} (redone: {changed})")
        fresh.release()

    check("first frame", True)
    check("nothing changed", False)
    world.step(1 / 60)
    check("a step", False)
    tile = still_tile(world)
    tile.material, tile.material_over = "glow", {}  # a glow with no colour of its own takes its fill colour's hue
    check("material assigned", True)
    tile.fill_col = [20, 240, 20]
    check("fill colour assigned (glow hue)", True)
    tile.material_over = {"emissive_strength": 2.0}
    check("material_over assigned", True)
    if getattr(world, "palette", None) is not None:
        palettes.apply_palette(world, [p for p in palettes.PALETTES if p != world.palette][0])
        check("palette switch", True)
    world.add(SoftBody(polygon_points(WORLD[0] / 2, 80, 20, 6), stiffness=0.5, point_radius=2))
    check("body added", True)
    gw.release()


def main(argv):
    EnvConfig.backend = "cpu"
    for name in argv or SCENES:
        run(name)
    materials_check()
    print(f"{bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
