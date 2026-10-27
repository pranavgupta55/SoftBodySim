"""Checks for the lit G-buffer writers of gl_world (research/lighting_contract.md) on a standalone context. Run from
the repo root:

    .venv/bin/python3 tests/gbuf_behind_check.py

- the behind layer: an opaque red rect under a glass disc shows in gbuf["behind"] (with its flags and material id)
  where the glass covers it, the glass itself is in gbuf["albedo"] there, an opaque body drawn after the glass stays
  on top, and with no GLASS / FLUID body (or in the points view) there are no "behind" keys at all
- outlines: a filled body's black border is an occluder without FLAG_LIT (and has FLAG_BORDER)
- points view: the body's silhouette (background between its sticks) carries the occluder flags, the sticks and dots
  themselves don't, and nothing is GLASS-kind there
Exit status 1 on the first failed check.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

import materials  # noqa: E402
from entities import FixedBody, polygon_points, rect_points  # noqa: E402
from gl_world import GLWorld  # noqa: E402
from solver import World  # noqa: E402

W, H = 400, 300
BG = (40, 44, 60)
OCC, CASTS, LIT, BORDER = 1, 2, 4, 8
RED, BLUE = (220, 30, 30), (30, 60, 220)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def body(pts, col, preset, **over):
    b = FixedBody(pts, point_col=col, stick_col=col, fill_col=col, point_radius=2)
    b.material, b.material_over = preset, dict(over)
    return b


def world(*bodies):
    w = World()
    w.background, w.lighting, w.caustics, w.lights = list(BG), True, False, []
    w.add_many(list(bodies))
    return w


def read(tex, comps=4, dtype=np.uint8):
    return np.frombuffer(tex.read(), dtype).reshape(H, W, comps) if comps > 1 else \
        np.frombuffer(tex.read(), dtype).reshape(H, W)


def flags(tex):
    return np.round(read(tex)[..., 3].astype(np.float32)).astype(np.int64)


def main():
    ctx = moderngl.create_standalone_context(require=410)
    gw = GLWorld(ctx, (W, H))
    render = lambda w, filled=True: gw.render(w, (W, H), filled, 1.0, BG, lighting=True)

    # red rect at x 100..200, glass disc centred (200, 150) r 60 drawn after it, blue rect (on top) at x 240..300
    red = body(rect_points(150, 150, 100, 120), RED, "matte", bevel_px=0, tint_strength=0)
    glass = body(polygon_points(200, 150, 60, 32), (200, 220, 240), "glass")
    blue = body(rect_points(270, 150, 60, 60), BLUE, "matte", bevel_px=0, tint_strength=0)
    w = world(red, glass, blue)
    render(w)
    gb = gw.gbuf
    check(all(k in gb for k in ("behind", "behind_nrm", "behind_mat")), "glass on screen: behind keys present")
    behind, albedo = read(gb["behind"]), read(gb["albedo"])
    p = (170, 150)  # x, y: inside the red rect and the glass disc
    check(np.abs(behind[p[1], p[0], :3].astype(int) - RED).max() <= 2, f"behind holds the red rect under the glass "
          f"({behind[p[1], p[0], :3]})")
    check(np.abs(albedo[p[1], p[0], :3].astype(int) - RED).max() > 40, "albedo holds the glass there")
    bmat, gmat = read(gb["behind_mat"], 1, np.uint16), read(gb["mat"], 1, np.uint16)
    check(bmat[p[1], p[0]] == gw.body_mat[0], "behind_mat: the red rect's material id")
    check(gmat[p[1], p[0]] == gw.body_mat[1], "mat: the glass's material id")
    check(flags(gb["behind_nrm"])[p[1], p[0]] & OCC, "behind_nrm: the red rect is an occluder")
    q = (250, 150)  # the blue rect over the glass disc (later in draw order)
    check(np.abs(albedo[q[1], q[0], :3].astype(int) - BLUE).max() <= 2, "opaque body later in order stays on top")
    check(gmat[q[1], q[0]] == gw.body_mat[2], "... with its own material id")
    bg_px = (20, 20)
    check(np.abs(behind[bg_px[1], bg_px[0], :3].astype(int) - BG).max() <= 1, "behind: the background elsewhere")
    tex = read(gw.tex)
    check(np.isfinite(tex).all(), "lit frame with the behind layer renders")

    # outlines: black border pixels are occluders, not lit, flagged border
    f = flags(gb["normal"])
    m = gmat
    top_edge = (150, 150 - 60)  # the red rect's top edge row (draw_border's 1 px outline at y = 90)
    fe = f[top_edge[1], top_edge[0]]
    check(fe & OCC and fe & BORDER and not fe & LIT, f"border: occluder + FLAG_BORDER, no FLAG_LIT (flags {fe})")
    check(f[150, 130] & LIT, "the body's fill stays lit")

    # no see-through body: one pass, no behind keys
    w2 = world(body(rect_points(150, 150, 100, 120), RED, "matte"))
    render(w2)
    check(not any(k in gw.gbuf for k in ("behind", "behind_nrm", "behind_mat")), "no glass / fluid: no behind keys")
    render(w)
    check("behind" in gw.gbuf, "glass again: behind back")
    w.bodies[1].material = "matte"
    render(w)
    check("behind" not in gw.gbuf, "glass turned matte: behind gone")
    w.bodies[1].material = "water"
    render(w)
    check(materials.of_body(w.bodies[1]).kind == materials.FLUID and "behind" in gw.gbuf, "fluid: behind present")

    # points view: silhouettes are occluders, sticks / dots are not, no glass kind anywhere, no behind keys
    ring = body(polygon_points(200, 150, 80, 24), RED, "matte")
    ring.point_radius = 7  # dots reaching well past the filled view's border
    gl2 = body(rect_points(80, 150, 60, 60), (200, 220, 240), "glass")
    w3 = world(ring, gl2)
    render(w3, filled=False)
    gb = gw.gbuf
    check("behind" not in gb, "points view: no behind keys (twins are opaque)")
    f, m, alb = flags(gb["normal"]), read(gb["mat"], 1, np.uint16), read(gb["albedo"])
    check(f[150, 200] & OCC and f[150, 200] & CASTS, "points view: inside the ring's silhouette is an occluder")
    check(m[150, 200] == 0 and np.abs(alb[150, 200, :3].astype(int) - BG).max() <= 1,
          "points view: ... while albedo / material stay the background")
    check(not f[20, 20] & OCC, "points view: background outside stays clear")
    kinds = gw.mat_kinds
    ids = np.unique(m[m > 0])
    check(ids.size > 0 and not np.isin(kinds[ids], (materials.GLASS, materials.FLUID)).any(),
          "points view: dots and sticks carry opaque twin materials")
    # a dot outside the silhouette: the ring's points are on its outline; a dot's outer half is outside it
    xs = np.nonzero((m[150] > 0))[0]
    outer = xs.max()  # the rightmost dot pixel on the middle row (outside the polygon's x range 280)
    check(outer >= 285 and not f[150, outer] & OCC, f"points view: dot pixel outside the outline is no occluder "
          f"(x {outer}, flags {f[150, outer]})")
    check(not f[150, 80] & OCC, "points view: a glass body's silhouette is no occluder")
    render(w3, filled=True)
    ff = flags(gw.gbuf["normal"])
    check(ff[150, 200] & OCC and ff[150, 200] & CASTS, "filled view: same silhouette occluder")
    gw.release()
    ctx.release()
    print("all passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL", e)
        sys.exit(1)
