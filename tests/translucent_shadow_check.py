"""Translucent shadows (gl_sdf's sdf_opaque / sdf_tr / sdf_tint, sdf_shadow_tr, light_accum's LIGHT_TR variant) and
GI through translucent bodies, on real GLWorld renders (standalone context). Run from the repo root:

    .venv/bin/python3 tests/translucent_shadow_check.py

A directional light straight down (dir (0, 1)) over flat background; a body at y 80..130 shadows the ground below
it. Light is read from LightsEffect's shared["light"] (diffuse, before compose), the final colour from GLWorld.tex.
(a) a red jelly (transparency 0.7) casts a reddish shadow, brighter than an opaque body's
(b) transparency 0 (glass): no translucent fields, and the light everywhere off the body is bit-identical to an
    opaque body's (today's shadows)
(c) a flat glass slab (transparency 0.8, absorption 0): caustics on -> the shadow is opaque and lit only by the
    caustics (no brighter than open ground + 5%); caustics off -> about transparency x open ground
(d) a water pool's shadow is lighter than an opaque one's
(e) GI: an emitter behind a clear glass slab lights the far side (an opaque slab blocks it)
(e2) GI energy: past fully clear glass panes the GI never exceeds the empty room's
(f) the points view: no translucent fields (see-through silhouettes are left out there)
Exit status 1 on the first failed check.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

import gl_lights  # noqa: E402
import gl_sdf  # noqa: E402
from entities import FixedBody, rect_points  # noqa: E402
from gl_world import GLWorld  # noqa: E402
from lights import DIRECTIONAL, Light  # noqa: E402
from solver import World  # noqa: E402

W, H = 400, 300
BG = (150, 150, 150)
SLAB = (200, 105, 160, 50)  # cx, cy, w, h: y 80..130
UNDER = (slice(190, 230), slice(185, 216))  # rows, cols: ground well inside the shadow
OPEN = (slice(190, 230), slice(40, 71))  # open ground, same rows


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def body(col, preset, rect=SLAB, **over):
    b = FixedBody(rect_points(*rect), point_col=col, stick_col=col, fill_col=col, point_radius=2)
    b.material, b.material_over = preset, dict(over)
    return b


def world(*bodies, caustics=False, light=True):
    w = World()
    w.background, w.lighting, w.caustics = list(BG), True, caustics
    w.ambient = (0.1, 0.1, 0.1)
    w.lights = [Light(kind=DIRECTIONAL, pos=(200.0, 10.0), dir=(0.0, 1.0), color=(1.0, 1.0, 1.0), intensity=1.0,
                      softness=1.0, height=600.0)] if light else []
    w.add_many(list(bodies))
    return w


class Renderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.gw = GLWorld(ctx, (W, H))

    def render(self, w, filled=True):
        self.gw.render(w, (W, H), filled, 1.0, BG, lighting=True)
        self.ctx.finish()
        return self.gw.frame.shared

    def light(self):
        tex = next(e for e in self.gw.effects if type(e).__name__ == "LightsEffect").light
        dt = np.float16 if tex.dtype == "f2" else np.float32
        return np.frombuffer(tex.read(), dt).reshape(H, W, 4)[..., :3].astype(np.float64)

    def image(self):
        return np.frombuffer(self.gw.tex.read(), np.uint8).reshape(H, W, 4)[..., :3].astype(np.float64)

    def gi(self):
        tex = self.gw.frame.shared["gi"]
        return np.frombuffer(tex.read(), np.float16).reshape(H, W, 4)[..., :3].astype(np.float64)


def mean(a, region):
    return a[region].reshape(-1, 3).mean(axis=0)


def main():
    ctx = moderngl.create_standalone_context(require=410)
    r = Renderer(ctx)
    red = (230, 40, 40)

    # opaque reference
    r.render(world(body(red, "matte")))
    opq_light = r.light()
    opq_under, open_l = mean(opq_light, UNDER), mean(opq_light, OPEN)
    check(open_l.mean() > 0.5, f"open ground is lit ({open_l})")
    check(opq_under.max() < 0.02 * open_l.mean(), f"an opaque body's shadow is dark ({opq_under})")

    # (a) red jelly
    sh = r.render(world(body(red, "jelly", transparency=0.7)))
    check(all(k in sh for k in ("sdf_opaque", "sdf_tr", "sdf_tint")), "jelly on screen: translucent fields")
    jl = mean(r.light(), UNDER)
    check(jl.mean() > 0.15 * open_l.mean(), f"(a) the jelly's shadow lets light through ({jl} vs open {open_l})")
    check(jl[0] > 1.3 * max(jl[1], jl[2]), f"(a) ... reddish ({jl})")
    check(jl[0] < 0.7 * open_l[0] + 1e-3, f"(a) ... but still a shadow ({jl[0]:.3f} < 0.7 x {open_l[0]:.3f})")

    # (b) transparency 0: today's shadow, bit for bit
    sh = r.render(world(body(red, "glass", transparency=0.0)))
    check("sdf_tr" not in sh, "(b) transparency 0: no translucent fields")
    g0 = r.light()
    off = np.ones((H, W), bool)
    x0, y0 = SLAB[0] - SLAB[2] // 2 - 3, SLAB[1] - SLAB[3] // 2 - 3
    off[y0:y0 + SLAB[3] + 6, x0:x0 + SLAB[2] + 6] = False
    check(np.array_equal(g0[off], opq_light[off]), "(b) ... light off the body identical to an opaque body's")

    # (c) a flat glass slab, absorption 0
    slab = dict(transparency=0.8, absorption=0.0)
    sh = r.render(world(body((200, 220, 240), "glass", **slab), caustics=False))
    check("sdf_tr" in sh, "(c) caustics off: translucent fields")
    sl = mean(r.light(), UNDER)
    ratio = sl.mean() / open_l.mean()
    check(abs(ratio - 0.8) < 0.08, f"(c) caustics off: light under the slab = {ratio:.3f} x open (transparency 0.8)")
    sh = r.render(world(body((200, 220, 240), "glass", **slab), caustics=True))
    check("sdf_tr" not in sh, f"(c) caustics on (CAUSTIC_SHARE {gl_lights.CAUSTIC_SHARE}): no translucent fields")
    img = r.image()
    lum = lambda a: a @ np.array([0.2126, 0.7152, 0.0722])  # noqa: E731
    under, opn = lum(mean(img, UNDER)), lum(mean(img, OPEN))
    check(under <= 1.05 * opn, f"(c) caustics on: under the slab {under:.1f} <= open {opn:.1f} + 5% (no double light)")
    check(mean(r.light(), UNDER).max() < 0.02 * open_l.mean(), "(c) ... its direct light is shadowed")

    # (d) water
    r.render(world(body((60, 120, 220), "water")))
    wl = mean(r.light(), UNDER)
    check(wl.mean() > 0.2 * open_l.mean() and wl.mean() > opq_under.mean() + 0.1,
          f"(d) a water pool's shadow is lighter than an opaque one's ({wl} vs {opq_under})")

    # (e) GI through glass: an emitter left of a tall slab, GI read right of it
    gi_vals = {}
    for name, preset, over in (("glass", "glass", dict(transparency=0.9, absorption=0.0)), ("opaque", "matte", {})):
        lamp = body((255, 200, 120), "glow", rect=(120, 150, 30, 30), emissive_strength=3.0)
        wall = body((200, 220, 240), preset, rect=(200, 150, 16, 240), **over)
        w = world(lamp, wall, light=False)
        w.gi = True
        r.render(w)
        gi_vals[name] = mean(r.gi(), (slice(140, 160), slice(230, 250))).mean()
    check(gi_vals["glass"] > 3.0 * gi_vals["opaque"] + 1e-4,
          f"(e) GI passes clear glass ({gi_vals['glass']:.4f}) but not an opaque wall ({gi_vals['opaque']:.4f})")

    # (e2) GI energy: past a pane of fully clear glass (transparency 1, no absorption, no emission) the GI is the
    # empty room's, never more (no light counted twice through glass)
    far = {}
    for name, panes in (("empty", ()), ("pane", ((200, 16),)), ("panes", tuple((180 + 10 * i, 4) for i in range(5)))):
        lamp = body((255, 200, 120), "glow", rect=(120, 150, 30, 30), emissive_strength=3.0)
        walls = [body((200, 220, 240), "glass", rect=(x, 150, wd, 240), transparency=1.0, absorption=0.0,
                      emissive_strength=0.0) for x, wd in panes]
        w = world(lamp, *walls, light=False)
        w.gi = True
        r.render(w)
        far[name] = mean(r.gi(), (slice(120, 180), slice(230, 360))).mean()
    for name in ("pane", "panes"):
        check(far[name] <= 1.01 * far["empty"] + 1e-5,
              f"(e2) GI past clear glass ({name}: {far[name]:.4f}) <= the empty room's ({far['empty']:.4f})")
    check(far["pane"] >= 0.95 * far["empty"], f"(e2) ... and one clear pane keeps it ({far['pane']:.4f})")

    # (f) points view
    sh = r.render(world(body(red, "jelly", transparency=0.7)), filled=False)
    check("sdf_tr" not in sh, "(f) points view: no translucent fields")
    print("all translucent shadow checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
