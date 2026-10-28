"""The static SDF cache (gl_sdf.SDF_CACHE): SdfEffect reuses its fields while every shadow caster is a fixed body
and nothing they depend on changed.
(a) fixed casters + a moving glowing body (casts no shadow, kept off the casters): the fields are reused from the
    second frame on, and every frame is byte for byte the uncached one (a lit, shadow-casting light: the field shows)
(b) a fixed caster moved (an editor drag): rebuilt, and that frame is the uncached one
(c) a material change, a colour change, a view switch, a zoom change, a world swap: rebuilt
(d) a moving body that casts shadows in the world: never reused
(e) the translucent fields (glass + a shadow-casting light) are reused along with shared["sdf"]: frames identical
Exit status 1 on the first failed check.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

import gl_sdf  # noqa: E402
from entities import FixedBody, RigidBody, polygon_points, rect_points  # noqa: E402
from gl_world import GLWorld  # noqa: E402
from lights import Light  # noqa: E402
from solver import World  # noqa: E402

W, H = 400, 300
BG = (40, 40, 50)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def body(cls, col, preset, pts, **over):
    b = cls(pts, point_col=col, stick_col=col, fill_col=col, point_radius=2)
    b.material, b.material_over = preset, dict(over)
    return b


def world(moving_caster=False):
    w = World()
    w.background, w.lighting = list(BG), True
    w.ambient = (0.05, 0.05, 0.05)
    w.lights = [Light(pos=(60.0, 60.0), radius=600.0, softness=4.0)]
    w.add_many([body(FixedBody, (200, 200, 200), "matte", rect_points(150, 150, 30, 60)),
                body(FixedBody, (200, 220, 240), "glass", rect_points(250, 120, 20, 80), transparency=0.6),
                body(FixedBody, (120, 90, 60), "matte", rect_points(300, 230, 60, 20)),
                body(RigidBody, (255, 200, 120), "glow", polygon_points(80, 240, 14, 6), emissive_strength=2.0)]
               + ([body(RigidBody, (90, 90, 90), "matte", polygon_points(330, 60, 12, 5))] if moving_caster else []))
    return w


class R:
    def __init__(self, ctx):
        self.ctx, self.gw = ctx, GLWorld(ctx, (W, H))

    def frame(self, w, filled=True, zoom=1.0):
        self.gw.render(w, (W, H), filled, zoom, BG, lighting=True)
        self.ctx.finish()
        return np.frombuffer(self.gw.tex.read(), np.uint8).reshape(H, W, 4).copy()

    def cached(self):
        return next(e for e in self.gw.effects if type(e).__name__ == "SdfEffect").stats.get("cached")


def shift(w, k, dx):
    """Move body k's points (a FixedBody: as an editor drag does) by dx."""
    o = w.offsets
    w.P[0, o[k]:o[k + 1]] += dx


def main():
    ctx = moderngl.create_standalone_context(require=410)
    check(gl_sdf.SDF_CACHE, "the cache is on")
    r, ref = R(ctx), R(ctx)

    def same(w, tag, **kw):
        gl_sdf.SDF_CACHE = False
        want = ref.frame(w, **kw)
        gl_sdf.SDF_CACHE = True
        got = r.frame(w, **kw)
        check(np.array_equal(got, want), f"{tag}: identical to the uncached frame")
        return r.cached()

    a = world()
    flags = []
    for k in range(5):
        shift(a, 3, 4.0)  # the glow body moves (it casts no shadow)
        flags.append(same(a, f"(a) frame {k}"))
    check(not flags[0] and all(flags[1:]), f"(a) reused from the second frame on ({flags})")

    shift(a, 0, 6.0)
    check(not same(a, "(b) a fixed caster moved"), "(b) ... rebuilt")
    check(same(a, "(b) next frame"), "(b) ... then reused again")

    a.bodies[2].material_over = {"roughness": 0.5}
    check(not same(a, "(c) material change"), "(c) material change: rebuilt")
    same(a, "(c) settle")
    a.bodies[0].fill_col = (10, 200, 10)
    check(not same(a, "(c) colour change"), "(c) colour change: rebuilt")
    same(a, "(c) settle")
    check(not same(a, "(c) view switch", filled=False), "(c) view switch: rebuilt")
    check(same(a, "(c) points view, next frame", filled=False), "(c) ... points view reused")
    check(not same(a, "(c) zoom change", zoom=0.9), "(c) zoom change: rebuilt")
    b = world()
    check(not same(b, "(c) world swap"), "(c) world swap: rebuilt")

    d = world(moving_caster=True)
    flags = []
    for k in range(3):
        shift(d, 4, 3.0)
        flags.append(same(d, f"(d) frame {k}"))
    check(not any(flags), f"(d) a moving caster: never reused ({flags})")

    e = world()
    gl_sdf.SDF_CACHE = False
    ref.frame(e)
    gl_sdf.SDF_CACHE = True
    r.frame(e)
    fx = next(x for x in r.gw.effects if type(x).__name__ == "SdfEffect")
    check("sdf_tr" in r.gw.frame.shared, "(e) setup: the translucent fields are built")
    check(same(e, "(e) translucent fields") and "sdf_tr" in r.gw.frame.shared and fx.stats["cached"],
          "(e) ... reused along with shared['sdf']")
    print("all SDF cache checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
