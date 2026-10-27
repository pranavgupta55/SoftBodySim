"""Checks for lights.sanitize / active and gl_lights.pack_lights on malformed lights, plus a GLWorld frame lit by
them. Run from the repo root:

    .venv/bin/python3 tests/lights_sanitize_check.py

- sanitize never raises and returns finite values inside the documented clamps (NaN / inf / negative / None / str /
  2-tuple / RGBA colours, a zero dir, falloff 0, softness 1e6, cones out of order, ...);
- an unknown kind, a non-finite or malformed position (point / spot) and a disabled light are dropped;
- Light.from_dict coerces the same way and never drops a light; of_world's ambient is 3 finite comps in [0, 4];
- pack_lights gives finite rows only, softness at least gl_lights.MIN_SOFTNESS_PX;
- gl_sdf's GI gains are clamped to [0, 50], a non-finite one counts as 1;
- one GLWorld frame of the lighting demo with all of these lights: the light / spec buffers are finite and the image
  has no black block that the same frame with only the valid lights doesn't have.
Exit status 1 on a failure.
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402

import gl_lights  # noqa: E402
import gl_sdf  # noqa: E402
import lights  # noqa: E402
from lights import DIRECTIONAL, POINT, SPOT, Light  # noqa: E402

FAILS = []
CAUSTICS_TOO = False
NAN, INF = float("nan"), float("inf")


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def bad_lights():
    """(light, should_draw) pairs."""
    L = Light
    return [
        (L(color=(NAN, INF, -1.0)), True),
        (L(color=None), True),
        (L(color="red"), True),
        (L(color=0.5), True),
        (L(color=(1.0, 0.5)), True),
        (L(color=(1.0, 0.5, 0.2, 0.7)), True),
        (L(color=(1e9, 1e9, 1e9)), True),
        (L(intensity=NAN), True),
        (L(intensity=-5.0), True),
        (L(intensity=INF), True),
        (L(intensity="bright"), True),
        (L(intensity=None), True),
        (L(kind=SPOT, dir=(0.0, 0.0)), True),
        (L(kind=DIRECTIONAL, dir=(NAN, 1.0)), True),
        (L(kind=SPOT, dir=(3.0, 4.0)), True),
        (L(falloff=0.0), True),
        (L(falloff=1e9), True),
        (L(softness=1e6), True),
        (L(kind=DIRECTIONAL, softness=1e6), True),
        (L(softness=-3.0), True),
        (L(radius=-10.0), True),
        (L(radius=NAN), True),
        (L(radius=INF), True),
        (L(height=-1.0), True),
        (L(height=INF), True),
        (L(kind=SPOT, cone_inner=90.0, cone_outer=10.0), True),
        (L(kind=SPOT, cone_inner=-5.0, cone_outer=900.0), True),
        (L(kind=SPOT, cone_inner=NAN, cone_outer=INF), True),
        (L(casts_shadows="yes"), True),
        (L(kind=DIRECTIONAL, pos=(NAN, NAN)), True),  # a directional light has no position
        (L(kind="laser"), False),
        (L(kind=None), False),
        (L(pos=(NAN, 0.0)), False),
        (L(kind=SPOT, pos=(0.0, INF)), False),
        (L(pos=(1.0,)), False),
        (L(pos=None), False),
        (L(pos="here"), False),
        (L(enabled=False), False),
    ]


def in_clamps(s):
    """Every documented clamp of a sanitised light."""
    vals = [*s.pos, *s.dir, *s.color, s.intensity, s.radius, s.falloff, s.cone_inner, s.cone_outer, s.softness,
            s.height]
    ok = all(isinstance(v, float) and math.isfinite(v) for v in vals)
    ok &= len(s.color) == 3 and all(0.0 <= c <= lights.MAX_COLOR for c in s.color)
    ok &= 0.0 <= s.intensity <= lights.MAX_INTENSITY
    ok &= abs(math.hypot(*s.dir) - 1.0) < 1e-6
    ok &= s.radius >= 0.0
    ok &= lights.FALLOFF_RANGE[0] <= s.falloff <= lights.FALLOFF_RANGE[1]
    soft_max = lights.MAX_SOFTNESS if s.kind == DIRECTIONAL else min(0.5 * s.radius, lights.MAX_SOFTNESS)
    ok &= 0.0 <= s.softness <= soft_max + 1e-9
    ok &= 0.0 <= s.height <= lights.MAX_HEIGHT
    ok &= 0.0 <= s.cone_inner <= s.cone_outer <= 180.0
    ok &= isinstance(s.casts_shadows, bool) and s.enabled is True
    return bool(ok)


def sanitize_checks(cases):
    for lt, draws in cases:
        try:
            s = lights.sanitize(lt)
        except Exception as e:  # noqa: BLE001
            check(False, f"sanitize raised {type(e).__name__} on {lt}")
            continue
        label = {k: v for k, v in lt.__dict__.items() if v != getattr(Light(), k)} or "default"
        if draws:
            check(s is not None and in_clamps(s), f"sanitised within clamps: {label}")
        else:
            check(s is None, f"dropped: {label}")
    s = lights.sanitize(Light(color=(NAN, INF, -1.0)))
    check(s.color == (0.0, lights.MAX_COLOR, 0.0), f"colour NaN -> 0, inf -> max, negative -> 0 {s.color}")
    check(lights.sanitize(Light(color=0.5)).color == (0.5, 0.5, 0.5), "scalar colour repeated")
    check(lights.sanitize(Light(color=(1.0, 0.5))).color == (1.0, 0.5, 0.0), "short colour padded")
    check(lights.sanitize(Light(color=(1.0, 0.5, 0.2, 0.7))).color == (1.0, 0.5, 0.2), "RGBA colour cut")
    check(lights.sanitize(Light(kind=SPOT, dir=(0.0, 0.0))).dir == (0.0, 1.0), "zero dir -> (0, 1)")
    check(lights.sanitize(Light(kind=SPOT, dir=(3.0, 4.0))).dir == (0.6, 0.8), "dir normalised")
    check(lights.sanitize(Light(falloff=0.0)).falloff == 0.01, "falloff 0 -> 0.01")
    check(lights.sanitize(Light(softness=1e6, radius=400)).softness == 200.0, "softness <= radius / 2")
    check(lights.sanitize(Light(kind=DIRECTIONAL, softness=1e6)).softness == 400.0, "directional softness <= 400")
    check(lights.sanitize(Light(intensity="bright")).intensity == 1.0, "unconvertible -> the default")
    s = lights.sanitize(Light(kind=SPOT, cone_inner=90.0, cone_outer=10.0))
    check(s.cone_inner == 10.0 and s.cone_outer == 10.0, "cone inner <= outer")
    good = Light(kind=SPOT, pos=(10.0, 20.0), dir=(0.0, 1.0))
    check(lights.sanitize(good) == good, "a valid light is unchanged")
    check(lights.sanitize(object()) is None and lights.sanitize(None) is None, "not a light: dropped, no raise")

    # from_dict never drops (keeps a malformed pos for the loader to refuse); same coercion otherwise
    d = Light(color=(NAN, 2.0), intensity="x", dir=(0, 0), softness=1e6, pos=(1.0,)).to_dict()
    d["junk"] = 1
    lt = Light.from_dict(d)
    check(isinstance(lt, Light) and lt.color == (0.0, 2.0, 0.0) and lt.intensity == 1.0 and lt.dir == (0.0, 1.0)
          and lt.softness == 200.0 and lt.pos == (1.0,), f"from_dict coerces, keeps the light {lt}")
    check(Light.from_dict(Light().to_dict()) == Light(), "from_dict round trip")

    # world-level helpers
    world = SimpleNamespace(lights=[c[0] for c in cases], ambient=(NAN, INF, -2.0))
    act = lights.active(world)
    check(len(act) == sum(1 for _, draws in cases if draws) and all(in_clamps(s) for s in act),
          f"active: {len(act)} sanitised lights")
    amb = lights.of_world(world)[1]
    check(amb == (0.0, lights.MAX_AMBIENT, 0.0), f"ambient NaN / inf / negative clamped {amb}")
    check(lights.of_world(SimpleNamespace(ambient=0.5))[1] == (0.5, 0.5, 0.5), "scalar ambient")
    check(lights.of_world(SimpleNamespace(ambient=(0.1, 0.2)))[1] == (0.1, 0.2, 0.0), "short ambient padded")
    check(lights.of_world(SimpleNamespace(ambient="dark"))[1] == lights.DEFAULT_AMBIENT, "bad ambient: default")
    check(lights.of_world(SimpleNamespace(lights=None))[1] == lights.DEFAULT_AMBIENT, "no ambient: the default")

    # pack_lights
    for zoom in (0.5, 1.0, 3.0, NAN, 0.0):
        try:
            data = gl_lights.pack_lights([c[0] for c in cases], zoom)
            ok = data.shape == (len(act), 16) and np.isfinite(data).all() and \
                (data[:, 13] >= gl_lights.MIN_SOFTNESS_PX).all()
        except Exception as e:  # noqa: BLE001
            ok = False
            print("   ", type(e).__name__, e)
        check(ok, f"pack_lights at zoom {zoom}: finite, {len(act)} rows, softness >= {gl_lights.MIN_SOFTNESS_PX}")
    hard = gl_lights.pack_lights([Light(softness=0.0)], 1.0)
    check(hard[0, 13] == gl_lights.MIN_SOFTNESS_PX, "hard shadows get the anti-aliased minimum softness")
    huge = gl_lights.pack_lights([Light(pos=(1e300, 1e300), radius=1e300)], 1.0)
    check(np.isfinite(huge).all(), "float32 overflow stays finite")


def gain_checks():
    saved = gl_sdf.GI_GAIN
    try:
        for g, want in ((6.5, 6.5), (NAN, 1.0), (INF, 1.0), (-3.0, 0.0), (1e6, 50.0), ("x", 1.0), (None, 1.0),
                        (0.0, 0.0)):
            got = gl_sdf.gi_gain(SimpleNamespace(gi_gain=g))
            check(got == want, f"GI gain {g!r} -> {got} (want {want})")
        gl_sdf.GI_GAIN = 20.0
        check(gl_sdf.gi_gain(SimpleNamespace(gi_gain=6.5)) == 50.0, "effective GI gain (product) <= 50")
        gl_sdf.GI_GAIN = NAN
        check(gl_sdf.gi_gain(SimpleNamespace()) == 1.0, "non-finite GI_GAIN -> 1")
    finally:
        gl_sdf.GI_GAIN = saved


def black_blocks(img, ref, n=8):
    """n x n blocks all black (max < 3) in img but not in ref."""
    h, w = img.shape[0] // n * n, img.shape[1] // n * n

    def blocks(a):
        return (a[:h, :w].max(axis=2) < 3).reshape(h // n, n, w // n, n).all(axis=(1, 3))
    return int((blocks(img) & ~blocks(ref)).sum())


def render_checks(cases):
    import pygame
    pygame.init()
    import scene
    from gl_world import GLWorld
    ctx = moderngl.create_standalone_context(require=410)
    keep = ctx.framebuffer([ctx.texture((4, 4), 4)])
    W, H, zoom = 900, 584, 0.5
    world = scene.PRESETS["lighting demo"](W / zoom, H / zoom)
    # caustics off: CausticsEffect (gl_glass, batch B-W2) reads raw lights until it takes lights.active; run them
    # once that lands by setting CAUSTICS_TOO = True
    world.caustics = CAUSTICS_TOO
    gw = GLWorld(ctx, (W, H))
    bg = getattr(world, "background", None) or (20, 22, 30)
    base = list(getattr(world, "lights", None) or [])
    # spread the test lights over the view (positions in world units)
    spread = []
    for i, (lt, _) in enumerate(cases):
        if lt.kind in (POINT, SPOT) and isinstance(lt.pos, tuple) and lt.pos == Light().pos:
            lt = lt.copy(pos=((i % 8 + 0.5) * W / zoom / 8, (i // 8 + 0.5) * H / zoom / 5))
        spread.append(lt)
    out = {}
    for name, ls in (("valid", base + [lt for lt, draws in zip(spread, (d for _, d in cases)) if draws]),
                     ("all", base + spread)):
        world.lights = ls
        keep.use()
        gw.render(world, (W, H), True, zoom, bg, lighting=True)
        ctx.finish()
        out[name] = np.frombuffer(gw.tex.read(), np.uint8).reshape(H, W, 4)[..., :3].copy()
        for key in ("light", "spec"):
            t = gw.frame.shared[key]
            arr = np.frombuffer(t.read(), np.float32 if t.dtype == "f4" else np.float16)
            check(np.isfinite(arr).all(), f"{name}: shared['{key}'] finite ({t.dtype})")
    check((out["all"] == out["valid"]).all(), "the dropped lights change nothing")
    nb = black_blocks(out["all"], out["valid"])
    check(nb == 0, f"no new black blocks ({nb})")
    check(out["all"].mean() > 5, f"the frame isn't black (mean {out['all'].mean():.1f})")
    gw.release() if hasattr(gw, "release") else None


def main():
    cases = bad_lights()
    sanitize_checks(cases)
    gain_checks()
    render_checks(cases)
    print(f"{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
