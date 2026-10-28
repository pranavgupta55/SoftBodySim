"""See-through bodies as a medium in the GI (gl_sdf GI_GLASS_CHORD): an emitter on the left, a glass wall across
the whole height in the middle, the GI measured behind it (relative to no wall).
(a) a clear wall passes nearly everything, a wall of transparency T one reference chord thick passes less than T
    (oblique rays cross more) but well over T^2
(b) a wall two chords thick passes about the square of what one chord passes (Beer-Lambert), not the same
(c) two touching one-chord walls pass what one two-chord wall does (each body counts: they used to count once)
(d) a stack of many thin walls well past the cascade's step budget still dims the light (no leak: a ray out of steps
    still crosses the rest of the medium)
(e) an opaque wall blocks it
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
BG = (0, 0, 0)
C = gl_sdf.GI_GLASS_CHORD


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def body(col, preset, rect, **over):
    b = FixedBody(rect_points(*rect), point_col=col, stick_col=col, fill_col=col, point_radius=2)
    b.material, b.material_over = preset, dict(over)
    return b


def world(walls=(), preset="glass", **over):
    """walls: (x centre, width) of glass walls spanning the height."""
    w = World()
    w.background, w.lighting, w.gi = list(BG), True, True
    w.ambient, w.lights = (0.0, 0.0, 0.0), []
    over = dict(dict(transparency=0.5, absorption=0.0, emissive_strength=0.0, reflectivity=0.0), **over)
    w.add_many([body((255, 255, 255), "glow", (40, 150, 30, 280), emissive_strength=3.0)]
               + [body((255, 255, 255), preset, (x, 150, wd, H + 40), **over) for x, wd in walls])
    return w


def gi_behind(ctx, w):
    gw = GLWorld(ctx, (W, H))
    for _ in range(3):
        gw.render(w, (W, H), True, 1.0, BG, lighting=True, outlines=False)
    ctx.finish()
    g = np.frombuffer(gw.frame.shared["gi"].read(), np.float16).reshape(gw.frame.shared["gi"].size[1], -1, 4)
    v = float(g[100:200, 330:370, :3].astype(np.float64).mean())
    gw.release()
    return v


def main():
    ctx = moderngl.create_standalone_context(require=410)
    open_ = gi_behind(ctx, world())
    check(open_ > 0, f"setup: GI behind with no wall ({open_:.4f})")
    r = {k: gi_behind(ctx, world(v)) / open_ for k, v in {
        "one": [(200, C)], "two": [(200, 2 * C)], "pair": [(200 - C / 2, C), (200 + C / 2, C)],
        "stack": [(150 + 4 * k, 2) for k in range(30)]}.items()}
    r["clear"] = gi_behind(ctx, world([(200, C)], transparency=1.0)) / open_
    r["opaque"] = gi_behind(ctx, world([(200, C)], preset="matte")) / open_
    print("   ratios:", {k: round(v, 3) for k, v in r.items()})
    check(r["clear"] > 0.9, f"(a) a clear wall passes nearly all ({r['clear']:.3f})")
    check(0.25 < r["one"] < 0.5, f"(a) one chord at T 0.5 passes under 0.5, over 0.25 ({r['one']:.3f})")
    check(abs(r["two"] - r["one"] ** 2) < 0.35 * r["one"] ** 2,
          f"(b) two chords ~ one squared ({r['two']:.3f} vs {r['one'] ** 2:.3f})")
    check(abs(r["pair"] - r["two"]) < 0.25 * r["two"], f"(c) two touching walls ~ one twice as thick "
          f"({r['pair']:.3f} vs {r['two']:.3f})")
    # 30 walls of 2 at T 0.5 per chord C: ~T^(60 / C) of what goes straight through, far below one wall's
    check(r["stack"] < 0.5 * r["two"], f"(d) a stack past the step budget still dims ({r['stack']:.3f})")
    check(r["opaque"] < 0.02, f"(e) an opaque wall blocks ({r['opaque']:.3f})")
    print("all GI medium checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
