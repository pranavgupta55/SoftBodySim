"""The GI far-field cache (gl_sdf.GI_FAR_CASCADES / GI_FAR_EVERY): the top cascades redone every GI_FAR_EVERY
frames and reused in between.
(a) a still scene renders byte for byte as with every cascade every frame, on refresh and on reuse frames alike
(b) the far cascades alternate: redone, reused, redone, ...
(c) a world swap redoes them at once, and that frame is the new world's own (= rendered without the cache)
(d) so do a view-mode switch, a zoom change, the lighting switched off and on, a material / colour change and a
    palette change
Exit status 1 on the first failed check.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

import gl_sdf  # noqa: E402
from entities import FixedBody, rect_points  # noqa: E402
from gl_world import GLWorld  # noqa: E402
from solver import World  # noqa: E402

W, H = 400, 300
BG = (40, 40, 50)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def body(col, preset, rect, **over):
    b = FixedBody(rect_points(*rect), point_col=col, stick_col=col, fill_col=col, point_radius=2)
    b.material, b.material_over = preset, dict(over)
    return b


def world(lamp_x=120, glass_col=(200, 220, 240)):
    w = World()
    w.background, w.lighting, w.gi = list(BG), True, True
    w.ambient = (0.05, 0.05, 0.05)
    w.lights = []
    w.add_many([body((255, 200, 120), "glow", (lamp_x, 150, 30, 30), emissive_strength=3.0),
                *[body(glass_col, "glass", (200 + 24 * k, 150, 16, 200), transparency=0.6, absorption=0.0)
                  for k in range(4)],
                body((90, 90, 90), "matte", (330, 60, 40, 40))])
    return w


class R:
    def __init__(self, ctx):
        self.ctx, self.gw = ctx, GLWorld(ctx, (W, H))

    def frame(self, w, filled=True, zoom=1.0, lighting=True):
        self.gw.render(w, (W, H), filled, zoom, BG, lighting=lighting)
        self.ctx.finish()
        return np.frombuffer(self.gw.tex.read(), np.uint8).reshape(H, W, 4).copy()

    def refreshed(self):
        return next(e for e in self.gw.effects if type(e).__name__ == "GiEffect").stats.get("far_refreshed")


def main():
    ctx = moderngl.create_standalone_context(require=410)
    every, n = gl_sdf.GI_FAR_EVERY, gl_sdf.GI_FAR_CASCADES
    check(every > 1 and n > 0, f"the cache is on (GI_FAR_EVERY {every}, GI_FAR_CASCADES {n})")
    # references: every cascade every frame
    gl_sdf.GI_FAR_EVERY = 1
    ref = R(ctx)
    a, b = world(), world(lamp_x=90, glass_col=(240, 200, 200))
    ref_a = [ref.frame(a) for _ in range(2)][-1]
    ref_b = [ref.frame(b) for _ in range(2)][-1]
    ref_a_pts = [ref.frame(a, filled=False) for _ in range(2)][-1]
    gl_sdf.GI_FAR_EVERY = every

    # (a) + (b)
    r = R(ctx)
    flags = []
    for k in range(6):
        img = r.frame(a)
        flags.append(r.refreshed())
        if k:  # (the very first lit frame of a GLWorld differs in the caustics path, not GI: compare from 1 on)
            check(np.array_equal(img, ref_a), f"(a) still scene, frame {k} ({'redone' if flags[-1] else 'reused'}):"
                  f" identical to every cascade every frame")
    check(flags[0] and any(not f for f in flags) and flags == [k % every == 0 for k in range(6)],
          f"(b) far cascades redone every {every} frames ({flags})")

    # (c) world swap right after a refresh frame (the next would reuse): redone, and the new world's own frame
    r.frame(a)
    check(r.refreshed(), "(c) setup: a refresh frame")
    img = r.frame(b)
    check(r.refreshed(), "(c) world swap: far cascades redone at once")
    check(np.array_equal(img, ref_b), "(c) ... the frame is the new world's own (= no cache)")

    # (d) the other whole-scene changes, each right after a refresh frame
    def after_refresh(w):
        while True:
            r.frame(w)
            if r.refreshed():
                return

    after_refresh(a)
    img = r.frame(a, filled=False)
    check(r.refreshed(), "(d) view switch (V): redone")
    r.frame(a, filled=False)
    img = r.frame(a, filled=False)
    check(np.array_equal(img, ref_a_pts), "(d) ... and the points view matches its no-cache render")
    after_refresh(a)
    r.frame(a, zoom=0.9)
    check(r.refreshed(), "(d) zoom change: redone")
    after_refresh(a)
    r.frame(a, lighting=False)
    a.lighting = True
    r.frame(a)
    check(r.refreshed(), "(d) lighting off and on: redone")
    after_refresh(a)
    a.bodies[1].material_over = dict(a.bodies[1].material_over, transparency=0.3)
    r.frame(a)
    check(r.refreshed(), "(d) a material change: redone")
    after_refresh(a)
    a.bodies[2].fill_col = (10, 200, 10)
    r.frame(a)
    check(r.refreshed(), "(d) a colour change: redone")
    after_refresh(a)
    a.palette = "some palette"
    r.frame(a)
    check(r.refreshed(), "(d) a palette change: redone")
    print("all GI far-cache checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
