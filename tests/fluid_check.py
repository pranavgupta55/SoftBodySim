"""The lit fluid filter (gl_glass's liquid resolve) on real GLWorld renders. Run from the repo root:

    .venv/bin/python3 tests/fluid_check.py [out_dir]

a. the liquid follows the light on the fluid itself: under a spot light the fluid preset's pool centre is at least
   30% brighter than with the spot off (zoom 0.5; the liquid is lit by the fluid bodies' own light ratio, not by the
   floor ratio inpainted mostly from the open floor above the pool; before: 1.21x).
b. no holes: at zoom 0.25 and 0.5 (the fluid preset, two point lights), no gap pixel inside the pool (a background
   pixel with fluid all around it) keeps the shadowed floor: each differs from it by more than 10% (in its most
   changed channel). And on the glass harness, with the bridge search cut to 1 px, no liquid bridge pixel whose
   search misses is left as the floor (they take the first fluid material).
c. small fluid bodies stay: an isolated fluid body of radius 4 px (zoom 1) and a 3 px wide stream show their own
   colour (bluer than the same render without them by > 0.03), not filtered away while their shadows stay.
With out_dir, writes the renders as PNGs. Exit status 1 on a failure.
"""
import os
import sys

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import pygame  # noqa: E402

pygame.init()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gl_glass  # noqa: E402
import materials  # noqa: E402
import presets_basic  # noqa: E402
from entities import FixedBody, polygon_points, rect_points  # noqa: E402
from gl_world import GLWorld  # noqa: E402
from lights import POINT, SPOT, Light  # noqa: E402
from looping import add_world_box  # noqa: E402
from solver import World  # noqa: E402

SIZE = W, H = 900, 584
FAILS = []
OUT = None  # main(): the PNG folder, if given


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def lum(img):
    return img[..., :3].astype(np.float64).mean(-1) / 255.0


def box_mean(mask, r):
    """Mean of `mask` over the (2r + 1)^2 box around each pixel (edges clamped), by a summed-area table."""
    m = np.pad(mask.astype(np.float64), r, mode="edge")
    s = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)


class Renderer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=410)
        self.gw = GLWorld(self.ctx, SIZE)

    def render(self, world, zoom, name=None):
        gw = self.gw
        bg = getattr(world, "background", None) or (40, 42, 56)
        gw.render(world, SIZE, True, zoom, bg, lighting=True, outlines=True)
        self.ctx.finish()
        img = np.frombuffer(gw.tex.read(), np.uint8).reshape(H, W, 4)[..., :3].copy()
        scene = np.frombuffer(gw.frame.shared["scene_lit"].read(), np.uint8).reshape(H, W, 4)[..., :3].copy()
        mat = np.frombuffer(gw.gbuf["mat"].read(), np.uint16).reshape(H, W)
        fluid = gw.mat_kinds[mat] == materials.FLUID
        if OUT and name:
            from PIL import Image
            Image.fromarray(img).save(os.path.join(OUT, f"fluid_{name}.png"))
        return img, scene, mat, fluid


def fluid_world(zoom, lights, steps=90):
    ww, wh = W / zoom, H / zoom
    world = presets_basic.build_fluid(ww, wh)
    world.background = [40, 42, 56]
    world.lighting, world.ambient, world.show_grid = True, (0.3, 0.3, 0.36), False
    for b in world.bodies:
        if getattr(b, "wall_side", None) is None:
            b.material, b.material_over = "water", {}
    world.lights = lights(ww, wh)
    for _ in range(steps):
        world.step(1 / 60)
    return world


def two_points(ww, wh):
    return [Light(kind=POINT, pos=(0.3 * ww, 0.35 * wh), color=(1, .85, .6), intensity=1.4, radius=.9 * ww, height=60),
            Light(kind=POINT, pos=(0.8 * ww, 0.4 * wh), color=(.5, .6, 1), intensity=1.0, radius=.7 * ww, height=50)]


def test_spot(r):
    zoom = 0.5
    blue = lambda ww, wh: Light(kind=POINT, pos=(0.1 * ww, 0.4 * wh), color=(.4, .6, 1), intensity=1.2,
                                radius=.7 * ww, height=50)
    spot = lambda ww, wh: Light(kind=SPOT, pos=(0.5 * ww, 0.3 * wh), dir=(0, 1), color=(1, .97, .9), intensity=2,
                                radius=1.0 * wh, cone_inner=10, cone_outer=20, height=80, softness=4)
    world = fluid_world(zoom, lambda ww, wh: [spot(ww, wh), blue(ww, wh)])
    on, _, _, fluid = r.render(world, zoom, "spot_on")
    world.lights = world.lights[1:]
    off, _, _, _ = r.render(world, zoom, "spot_off")
    # the pool's centre: fluid all around (box coverage), in the middle fifth of the view across
    inside = box_mean(fluid, 10) > 0.5
    cols = np.zeros_like(inside)
    cols[:, int(0.42 * W):int(0.58 * W)] = True
    sel = inside & cols
    a, b = lum(on)[sel].mean(), lum(off)[sel].mean()
    check(sel.sum() > 500 and a >= 1.3 * b,
          f"spot over the pool: centre {a:.3f} with the spot vs {b:.3f} without ({a / max(b, 1e-6):.2f}x, >= 1.3x; "
          f"{int(sel.sum())} px)")


def test_holes(r):
    for zoom in (0.25, 0.5):
        world = fluid_world(zoom, two_points)
        img, scene, mat, fluid = r.render(world, zoom, f"pool_z{zoom}")
        # gap pixels inside the pool: background with fluid all around (well inside the liquid), off the view's rim
        pool = box_mean(fluid, 8) > 0.5
        inside = (box_mean(pool, 6) > 0.999) & (box_mean(fluid, 3) > 0.35) & (mat == 0)  # 6 px in from its rim
        inside[:6], inside[-6:], inside[:, :6], inside[:, -6:] = False, False, False, False
        floor = scene[inside].astype(np.float64) / 255.0  # scene_lit there: the floor in the fluid's shadow
        out = img[inside].astype(np.float64) / 255.0
        rel = (np.abs(out - floor) / np.maximum(floor, 0.02)).max(-1)  # the most changed channel (a hue shift counts)
        n_bad = int((rel <= 0.1).sum())
        if OUT:
            from PIL import Image
            vis = img.copy()
            ys, xs = np.nonzero(inside)
            bad = rel <= 0.1
            vis[ys[bad], xs[bad]] = (255, 0, 255)
            Image.fromarray(vis).save(os.path.join(OUT, f"fluid_pool_z{zoom}_bad.png"))
        check(inside.sum() > 200 and n_bad == 0,
              f"no holes at zoom {zoom}: {n_bad} of {int(inside.sum())} gap px inside the pool within 10% of the "
              f"shadowed floor (min diff {rel.min() if len(rel) else 0:.2f})")


def search_hits(mat, x, y):
    """Whether glass_resolve's bridge search (8 rays, radii 1, 3, ... FLUID_SEARCH) from pixel (x, y) meets a body."""
    h, w = mat.shape
    for r in range(1, gl_glass.FLUID_SEARCH + 1, 2):
        for dx, dy in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
            if mat[min(max(y + dy * r, 0), h - 1), min(max(x + dx * r, 0), w - 1)] != 0:
                return True
    return False


def bilinear_half(field, h, w):
    """glass_resolve's half_at(field, p + 0.5) at every render px (h, w): bilinear, edges clamped."""
    fh, fw = field.shape
    ys, xs = (np.arange(h) + 0.5) / 2 - 0.5, (np.arange(w) + 0.5) / 2 - 0.5
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    c = lambda a, n: np.clip(a, 0, n - 1)
    f = lambda yy, xx: field[c(yy, fh)[:, None], c(xx, fw)[None, :]]
    return ((1 - fy) * (1 - fx) * f(y0, x0) + (1 - fy) * fx * f(y0, x0 + 1) + fy * (1 - fx) * f(y0 + 1, x0)
            + fy * fx * f(y0 + 1, x0 + 1))


def test_search_miss(ctx):
    """Liquid bridge pixels whose 8-ray search finds no fluid body are still liquid: rewritten, never left showing
    the floor. (With the real FLUID_SEARCH a miss needs a void the blur still calls liquid, which the fluid preset
    rarely makes; the search is cut to 1 px here so that most bridge pixels of a hex-packed pool miss.)"""
    import glass_harness as gh
    search, gl_glass.FLUID_SEARCH = gl_glass.FLUID_SEARCH, 1
    try:
        water = materials.PRESETS["water"]
        shapes = []
        R = 9.0  # hex packing like the fluid preset at zoom 0.5: octagons of r 7.5, 3 px gaps
        for row in range(6):
            for k in range(12):
                x, y = 350 + 2 * R * k + (R if row % 2 else 0), 250 + row * np.sqrt(3) * R
                shapes.append(gh.Shape(gh.ngon(x, y, 7.5, 8, 0.3 * k), water, (0.2, 0.5, 0.9)))
        data = gh.build(shapes, np.full((gh.SIZE[1], gh.SIZE[0], 3), 0.4))
        harness = gh.Harness(ctx, data, sdf=False)
        caus, glass = gl_glass.make(ctx, data["size"])
        harness.run([caus, glass])
        out = harness.read()[..., :3]
        field = harness.read(glass.pool.items["field_a"])[..., 0]
        h, w = data["mat"].shape
        liquid = bilinear_half(field, h, w) >= gl_glass.FLUID_THRESHOLD + 0.01  # (a margin for half-float rounding)
        ys, xs = np.nonzero(liquid & (data["mat"] == 0))
        miss = [(x, y) for x, y in zip(xs, ys) if not search_hits(data["mat"], x, y)]
        caus.release()
        glass.release()
    finally:
        gl_glass.FLUID_SEARCH = search
    kept = [(x, y) for x, y in miss if (out[y, x] == harness.scene_bytes[y, x, :3]).all()]
    check(len(miss) > 0 and not kept, f"bridge search misses: {len(kept)} of {len(miss)} liquid px whose search "
          f"finds no fluid body left as the floor (0; of {len(xs)} liquid bridge px)")


def drops_world(with_drops):
    zoom = 1.0
    world = World()
    add_world_box(world, W, H, 4.0, col=(60, 62, 70))
    world.background, world.lighting, world.ambient, world.show_grid = [60, 62, 76], True, (0.3, 0.3, 0.36), False
    world.lights = [Light(kind=POINT, pos=(0.35 * W, 0.3 * H), color=(1, .85, .6), intensity=1.4, radius=.9 * W,
                          height=60)]
    for wall in getattr(world, "walls", {}).values():
        wall.material, wall.material_over = "matte", {"casts_shadow": False}
    if with_drops:
        col = (40, 120, 200)
        for pts in (polygon_points(0.3 * W, 0.5 * H, 4.0, 8), rect_points(0.6 * W, 0.5 * H, 3.0, 60.0)):
            b = FixedBody(pts, point_col=col, stick_col=col, fill_col=col, point_radius=2, stick_width=1)
            b.material, b.material_over = "water", {}
            world.add(b)
    return world, zoom


def test_small(r):
    world, zoom = drops_world(True)
    img, _, mat, fluid = r.render(world, zoom, "drops")
    world0, _ = drops_world(False)
    ref, _, _, _ = r.render(world0, zoom, "drops_none")
    for name, x0, x1 in (("r=4 drop", 0.25, 0.35), ("3 px stream", 0.55, 0.65)):
        sel = fluid.copy()
        sel[:, :int(x0 * W)], sel[:, int(x1 * W):] = False, False
        # the body's own (blue) colour shows: blue minus red rises over the floor's (its shadow only darkens)
        diff = img.astype(np.float64) - ref
        d = float((diff[..., 2] - diff[..., 0])[sel].mean()) / 255.0 if sel.any() else 0.0
        check(sel.sum() >= 10 and d > 0.03, f"{name} visible: its {int(sel.sum())} px turn {d:.3f} bluer than the "
              f"floor on average (> 0.03)")


def main():
    global OUT
    OUT = sys.argv[1] if len(sys.argv) > 1 else None
    if OUT:
        os.makedirs(OUT, exist_ok=True)
    r = Renderer()
    test_spot(r)
    test_holes(r)
    test_search_miss(r.ctx)
    test_small(r)
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
