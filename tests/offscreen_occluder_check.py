"""The lit-render margin (gl_world.LIT_MARGIN): shadow casters just outside the view still shade it. Real GLWorld
renders on a standalone context; run from the repo root:

    .venv/bin/python3 tests/offscreen_occluder_check.py

A 400x300 view at zoom 1 with a wall at x -40..-10 (wholly off screen, left), rows 60..240. Light is read from
LightsEffect's shared["light"] (diffuse, before compose).
(a) a directional light travelling right: the wall shades the view's left edge in its rows (with LIT_MARGIN 0, the
    old behaviour, that edge is lit); rows above the wall stay lit and match the no-margin render
(b) a point light off screen past the wall: the wall shades the edge too
(c) the points view (filled=False): the same shadow (the margin draws the filled silhouettes)
(d) a clear glass wall instead (caustics off): a tinted, partial shadow, lighter than the opaque wall's
(e) the wall moved on screen: no margin (FrameInfo.occ None) and the render is bit-identical to LIT_MARGIN 0
(f) a caster that doesn't cast (casts_shadow False) off screen: no margin
(g) periodic axes get no margin (margin_limits), and a looping world renders without one
Exit status 1 on the first failed check.
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

import gl_world  # noqa: E402
from entities import FixedBody, rect_points  # noqa: E402
from gl_world import GLWorld  # noqa: E402
from lights import DIRECTIONAL, POINT, Light  # noqa: E402
from solver import World  # noqa: E402

W, H = 400, 300
BG = (150, 150, 150)
WALL = (-25, 150, 30, 180)  # cx, cy, w, h: x -40..-10, y 60..240 (off screen, left)
EDGE = (slice(110, 190), slice(2, 40))  # rows, cols: the view's left edge, in the wall's rows
ABOVE = (slice(5, 30), slice(2, 40))  # the left edge above the wall's rows
BELOW = (slice(262, 296), slice(2, 40))  # the left edge below the wall's rows (open ground)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def body(col, preset, rect=WALL, **over):
    b = FixedBody(rect_points(*rect), point_col=col, stick_col=col, fill_col=col, point_radius=2)
    b.material, b.material_over = preset, dict(over)
    return b


def directional():
    return Light(kind=DIRECTIONAL, pos=(200.0, 150.0), dir=(1.0, 0.0), color=(1.0, 1.0, 1.0), intensity=1.0,
                 softness=1.0, height=600.0)


def world(*bodies, lights=None):
    w = World()
    w.background, w.lighting, w.caustics = list(BG), True, False
    w.ambient = (0.1, 0.1, 0.1)
    w.lights = [directional()] if lights is None else lights
    w.add_many(list(bodies))
    return w


class Renderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.gw = GLWorld(ctx, (W, H))

    def render(self, w, filled=True, margin=128):
        gl_world.LIT_MARGIN = margin
        try:
            self.gw.render(w, (W, H), filled, 1.0, BG, lighting=True)
        finally:
            gl_world.LIT_MARGIN = 128
        self.ctx.finish()
        return self.gw.frame

    def light(self):
        tex = next(e for e in self.gw.effects if type(e).__name__ == "LightsEffect").light
        dt = np.float16 if tex.dtype == "f2" else np.float32
        return np.frombuffer(tex.read(), dt).reshape(H, W, 4)[..., :3].astype(np.float64)

    def image(self):
        return np.frombuffer(self.gw.tex.read(), np.uint8).reshape(H, W, 4)[..., :3].copy()


def mean(a, region):
    return float(a[region].reshape(-1, 3).mean())


def main():
    ctx = moderngl.create_standalone_context(require=410)
    r = Renderer(ctx)
    grey = (120, 120, 130)

    # (a) directional light, opaque wall off screen
    fr = r.render(world(body(grey, "stone")), margin=0)
    check(fr.occ is None, "(a) LIT_MARGIN 0: no margin")
    old = r.light()
    fr = r.render(world(body(grey, "stone")))
    check(fr.occ is not None and fr.origin[0] > 0, f"(a) a caster off screen: margin on (origin {fr.origin}, "
          f"occ {fr.occ['normal'].size if fr.occ else None})")
    check(fr.shared.get("sdf_origin") == (float(fr.origin[0]), float(fr.origin[1])),
          "(a) shared['sdf_origin'] is the margin's origin")
    new = r.light()
    open_l = mean(new, BELOW)
    check(open_l > 0.5, f"(a) below the wall's rows the edge is lit ({open_l:.3f})")
    check(mean(old, EDGE) > 0.5 * open_l, f"(a) without the margin the edge is lit ({mean(old, EDGE):.3f})")
    check(mean(new, EDGE) < 0.05 * open_l, f"(a) with it the wall shades the edge ({mean(new, EDGE):.3f})")
    check(abs(mean(new, ABOVE) - mean(old, ABOVE)) < 0.01 * open_l,
          f"(a) above the wall's rows: lit as before ({mean(new, ABOVE):.3f} vs {mean(old, ABOVE):.3f})")
    opaque_edge = mean(new, EDGE)

    # (b) a point light off screen, past the wall
    pt = Light(kind=POINT, pos=(-150.0, 150.0), color=(1.0, 1.0, 1.0), intensity=2.0, radius=900.0, falloff=0.5,
               softness=1.0, height=600.0)
    r.render(world(body(grey, "stone"), lights=[pt]), margin=0)
    old_pt = mean(r.light(), EDGE)
    r.render(world(body(grey, "stone"), lights=[pt]))
    new_pt = mean(r.light(), EDGE)
    check(old_pt > 0.1, f"(b) the off-screen point light reaches the edge without the margin ({old_pt:.3f})")
    check(new_pt < 0.05 * old_pt, f"(b) with it the wall shades the edge ({new_pt:.3f} vs {old_pt:.3f})")

    # (c) points view: the same shadow
    fr = r.render(world(body(grey, "stone")), filled=False)
    check(fr.occ is not None, "(c) points view: margin on")
    pv = r.light()
    check(mean(pv, EDGE) < 0.05 * open_l, f"(c) points view: the wall shades the edge ({mean(pv, EDGE):.3f})")

    # (d) a clear glass wall, caustics off: tinted, partial shadow
    fr = r.render(world(body((200, 220, 240), "glass", transparency=0.8, absorption=0.0)))
    check(all(k in fr.shared for k in ("sdf_opaque", "sdf_tr", "sdf_tint")), "(d) glass off screen: translucent "
          "fields (built over the margin)")
    gl = mean(r.light(), EDGE)
    check(opaque_edge + 0.2 * open_l < gl < 0.95 * open_l,
          f"(d) the glass wall's shadow is partial ({gl:.3f}; opaque {opaque_edge:.3f}, open {open_l:.3f})")

    # (e) the wall on screen: no margin, bit-identical
    inside = (25, 150, 30, 180)
    fr = r.render(world(body(grey, "stone", rect=inside)))
    check(fr.occ is None, "(e) every caster on screen: no margin")
    a = r.image()
    r.render(world(body(grey, "stone", rect=inside)), margin=0)
    check(np.array_equal(a, r.image()), "(e) ... and the render is bit-identical to LIT_MARGIN 0")

    # (f) an off-screen body that casts no shadow
    fr = r.render(world(body(grey, "stone", casts_shadow=False)))
    check(fr.occ is None, "(f) an off-screen body casting no shadow: no margin")

    # (g) periodic axes
    lim = GLWorld.margin_limits
    check(lim(SimpleNamespace(period=None)) == (128, 128), "(g) not periodic: margin on both axes")
    check(tuple(lim(SimpleNamespace(period=(800.0, 0.0)))) == (0, 128), "(g) periodic in x: none on x")
    check(tuple(lim(SimpleNamespace(period=(800.0, 600.0)))) == (0, 0), "(g) periodic on both: none")
    import scene
    lw = scene.PRESETS["looping"](W, H)
    for _ in range(5):
        lw.step(1 / 60)
    if lw.period is not None and all(float(p) > 0 for p in lw.period):
        fr = r.render(lw)
        check(fr.occ is None, "(g) the looping preset (periodic both ways) renders without a margin")
    else:
        print("skip (g) the looping preset isn't periodic on both axes yet:", lw.period)
    print("all off-screen occluder checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
