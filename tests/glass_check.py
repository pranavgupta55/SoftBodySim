"""Assertions for gl_glass (workstream G) on the synthetic harness (tests/glass_harness.py). Run from the repo root:

    .venv/bin/python3 tests/glass_check.py

1. refraction: inside a glass body the floor is sampled displaced (decoded from a coordinate-ramp floor): on the
   tilted rim by at least ~MIN_REFRACT_PX, toward the body's interior; with ior 1 it isn't displaced.
2. opaque pixels and the background stay byte-identical (away from the fluid filter's reach).
3. the fluid filter connects two touching fluid blobs (the gap between them turns liquid; with the filter off it
   doesn't), both with and without the SDF.
4. mirrors change their pixels (reflection), with and without the SDF march.
5. GlassEffect's floor redraw (GLWorld's base pass run again) gives exactly GLWorld's background.
Exit status 1 on a failure.
"""
import os
import sys
from dataclasses import replace

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402

import glass_harness as gh  # noqa: E402
import gl_glass  # noqa: E402
import materials  # noqa: E402
from materials import Material  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def run(ctx, data, fluid=True, **kw):
    h = gh.Harness(ctx, data, **kw)
    caus, glass = gl_glass.make(ctx, data["size"])
    glass.fluid_filter = fluid
    h.run([caus, glass])
    out = h.read()
    caus.release()
    glass.release()
    return out, h.scene_bytes


def ramp_floor(size):
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x % 256) / 255.0, (y % 256) / 255.0, np.full(x.shape, 0.5)], axis=-1)


def test_refraction(ctx):
    test_glass = Material(kind=materials.GLASS, ior=1.5, transparency=1.0, absorption=0.0, reflectivity=0.0,
                          specular=0.0, bevel_px=8.0, profile=materials.BEVEL)
    cx, cy, r = 450.0, 290.0, 60.0
    for ior in (1.5, 1.0):
        m = replace(test_glass, ior=ior)
        data = gh.build([gh.Shape(gh.ngon(cx, cy, r, 64), m, (0.9, 0.9, 0.9))], ramp_floor(gh.SIZE), light=False)
        for sdf in (False, True):
            out, _ = run(ctx, data, sdf=sdf)
            ys, xs = np.nonzero(data["mat"] > 0)
            ox = (out[ys, xs, 0].astype(int) - xs + 128) % 256 - 128
            oy = (out[ys, xs, 1].astype(int) - ys + 128) % 256 - 128
            n = data["normal"][ys, xs]
            rim = np.hypot(n[:, 0], n[:, 1]) > 0.3
            flat = data["dist"][ys, xs] > 12
            mag = np.hypot(ox, oy)
            inward = (ox * n[:, 0] + oy * n[:, 1]) / np.maximum(mag * np.hypot(n[:, 0], n[:, 1]), 1e-9)
            tag = f"ior {ior} sdf {sdf}"
            if ior > 1:
                check(np.median(mag[rim]) >= gl_glass.MIN_REFRACT_PX - 0.5,
                      f"refraction: rim offset median {np.median(mag[rim]):.2f} px >= ~MIN_REFRACT_PX ({tag})")
                check(np.median(inward[rim]) < -0.7,
                      f"refraction: rim samples bend toward the interior (cos {np.median(inward[rim]):.2f}) ({tag})")
            else:
                check(np.median(mag[rim]) <= 0.5, f"refraction: no offset with ior 1 ({np.median(mag[rim]):.2f} px)"
                      f" ({tag})")
            # ior 1.5 mixes in F0 = 4% Fresnel reflection of the blurred surroundings, which skews the decoded
            # coordinates by up to ~4% of the body's radius (2.4 px here) without displacing anything
            tol = 0.5 if ior == 1.0 else 0.5 + 0.04 * r
            check(np.median(mag[flat]) <= tol, f"refraction: flat interior undisplaced "
                  f"({np.median(mag[flat]):.2f} px <= {tol:.1f}) ({tag})")


def box_dilate(mask, r):
    """mask grown by r px (square), via an integral image."""
    h, w = mask.shape
    ii = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    return s > 0


def test_opaque_unchanged(ctx):
    data = gh.build(gh.demo_shapes(), gh.checker(gh.SIZE))
    kinds = np.array([-1] + [m.kind for m in data["mats"][1:]])[data["mat"]]
    special = (kinds == materials.GLASS) | (kinds == materials.MIRROR)
    fluid_reach = box_dilate(kinds == materials.FLUID, 24)
    keep = ~special & ~fluid_reach
    for sdf in (False, True):
        for behind in (True, False):
            out, scene = run(ctx, data, sdf=sdf, behind=behind)
            diff = np.any(out != scene, axis=-1)
            check(not diff[keep].any(), f"opaque / background pixels unchanged (sdf {sdf}, floor {behind}): "
                  f"{int(diff[keep].sum())} differ")
            check(diff[special].mean() > 0.5, f"glass / mirror pixels rewritten ({diff[special].mean():.0%}) "
                  f"(sdf {sdf}, floor {behind})")


def test_fluid_connects(ctx):
    water = materials.PRESETS["water"]
    r, gap = 7.5, 3.0  # the fluid preset at zoom 0.5: circumradius 7.5 px, 3 px between neighbours
    c0, c1 = np.array([440.0, 300.0]), np.array([440.0 + 2 * r + gap, 300.0])
    shapes = [gh.Shape(gh.ngon(*c, r, 8, np.pi / 8), water, (0.2, 0.5, 0.9)) for c in (c0, c1)]
    data = gh.build(shapes, gh.checker(gh.SIZE))
    gx = int(round((c0[0] + c1[0]) / 2))  # the gap's middle column
    ys = np.arange(296, 305)
    check(np.all(data["mat"][ys, gx] == 0), "fluid: the gap between the blobs is background in the G-buffer")
    for sdf in (False, True):
        out, scene = run(ctx, data, sdf=sdf)
        changed = np.any(out[ys, gx] != scene[ys, gx], axis=-1)
        check(changed.all(), f"fluid: the filter fills the gap between touching blobs (sdf {sdf}): "
              f"{int(changed.sum())}/{len(ys)} px")
        far = np.any(out[300, 400:420] != scene[300, 400:420], axis=-1)
        check(not far.any(), f"fluid: background well away from the blobs untouched (sdf {sdf})")
    out, scene = run(ctx, data, fluid=False, sdf=False)
    check(not np.any(out[ys, gx] != scene[ys, gx]), "fluid: without the filter the gap stays background")
    # the same pair across the x seam of a periodic map: connected only when the blur wraps
    w = gh.SIZE[0]
    c0 = np.array([w - r - gap / 2 - 0.5, 300.0])
    shapes = [gh.Shape(gh.ngon(*c, r, 8, np.pi / 8), water, (0.2, 0.5, 0.9)) for c in (c0, c0 + [2 * r + gap - w, 0])]
    data = gh.build(shapes, gh.checker(gh.SIZE))
    cols = [w - 2, w - 1, 0, 1]
    check(np.all(data["mat"][300, cols] == 0), "fluid: the gap across the seam is background in the G-buffer")
    for period, want in (((float(w), 0.0), True), (None, False)):
        out, scene = run(ctx, data, sdf=False, period=period)
        changed = np.any(out[300, cols] != scene[300, cols], axis=-1)
        check(changed.all() if want else not changed.any(),
              f"fluid: across the seam the gap {'fills' if want else 'stays open'} with period {period}")


def test_mirror(ctx):
    data = gh.build(gh.demo_shapes()[3:6], gh.checker(gh.SIZE))
    mirror = (data["mat"] > 0) & (np.array([-1] + [m.kind for m in data["mats"][1:]])[data["mat"]] == materials.MIRROR)
    for sdf in (False, True):
        out, scene = run(ctx, data, sdf=sdf)
        diff = np.abs(out[..., :3].astype(int) - scene[..., :3]).sum(-1)
        check((diff[mirror] > 6).mean() > 0.5, f"mirror reflects (sdf {sdf}): {(diff[mirror] > 6).mean():.0%} px "
              "changed")


def test_floor_redraw(ctx):
    import gl_world
    from entities import SoftBody, polygon_points
    from solver import World
    world = World()
    world.add_many([SoftBody(polygon_points(300, 300, 80, 8), stiffness=0.1, fill_col=(200, 60, 60),
                             point_col=(200, 60, 60), stick_col=(200, 60, 60))])
    size = (640, 480)
    bg = (30, 40, 60)
    gw = gl_world.GLWorld(ctx, size)
    target = ctx.simple_framebuffer(size)  # GLWorld.render restores the current framebuffer: give it a live one
    target.use()
    glass = gl_glass.GlassEffect(ctx, size)
    frame = type("F", (), {"gbuf": {}})()
    read = lambda t: np.frombuffer(t.read(), np.uint8).reshape(size[1], size[0], 4)[..., :3].copy()
    # no grid: the redrawn floor is exactly the world drawn without its bodies
    gw.render(World(), size, True, 0.5, bg)
    ref = read(gw.tex)
    gw.render(world, size, True, 0.5, bg)
    check(np.any(read(gw.tex) != ref), "floor redraw: the body shows in GLWorld.tex")
    check(np.array_equal(read(glass._floor(gw, frame)), ref), "floor redraw: = the world drawn without bodies")
    # grid on (its cells light up under the body): the redraw matches GLWorld.tex everywhere off the body
    gw.render(world, size, True, 0.5, bg, grid={})
    with_body = read(gw.tex)
    got = read(glass._floor(gw, frame))
    off = np.ones(size[::-1], bool)
    off[150 - 46:150 + 47, 150 - 46:150 + 47] = False  # the body: 80 world units at zoom 0.5 around (150, 150) px
    check(np.array_equal(got[off], with_body[off]) and np.any(got[~off] != with_body[~off]),
          "floor redraw with the grid: = GLWorld.tex off the body, the floor (not the body) under it")
    glass.release()


def main():
    ctx = moderngl.create_standalone_context(require=410)
    test_refraction(ctx)
    test_opaque_unchanged(ctx)
    test_fluid_connects(ctx)
    test_mirror(ctx)
    test_floor_redraw(ctx)
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
