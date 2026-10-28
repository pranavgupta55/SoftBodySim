"""Assertions for the caustics pass (gl_glass.CausticsEffect: caustics_rays.geom, glass_common.march, caustics_blur,
caustics_normals) on the synthetic harness (tests/glass_harness.py). Run from the repo root:

    .venv/bin/python3 tests/caustics_check.py

a. a lens under a DIRECTIONAL light (dir (0, 1)) focuses light below it (directional rays start where they enter the
   view, not half a diagonal outside it);
b. a stack of 3 lenses still deposits light below the last one (> 25% of a single lens's peak): caustic rays have
   their own interaction budget (MARCH_BUDGET), 6 refractions don't use it up;
c. the glass view / resolve output is byte-identical with the glass_common.glsl of BASELINE_REV (the caustics-only
   changes are isolated behind CAUSTIC_PIECES). Skipped without git;
d. the deposit is soft-capped (no channel above CAUSTIC_CAP * 1.05 at a hard focus), and a beam cut by the path cap
   (CAUSTIC_MAX_T) fades out (no step > 10/255 along it) instead of ending in a hard edge;
e. a light inside a glass disc (at its centre: every ray leaves along the normal) lights the ring around the disc
   evenly, not as a starburst;
f. a 1 px glass sliver in a beam doesn't fan the light into streaks: what crosses it goes on unbent (the deposit
   behind it is even along the sliver);
h. on periodic maps directional rays light each pixel once (no doubled strips from wrapped copies);
i. lights with NaN / infinite position, direction or falloff draw nothing and leave the deposit finite.
Exit status 1 on a failure.
"""
import os
import re
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import moderngl  # noqa: E402

import glass_harness as gh  # noqa: E402
import gl_glass  # noqa: E402
import lights  # noqa: E402
import materials  # noqa: E402
from materials import Material  # noqa: E402

# glass_common.glsl before the caustics-only changes (bump it when the view / resolve march changes on purpose)
BASELINE_REV = "43b1fca"
FAILS = []
W, H = gh.SIZE
ZOOM = gh.ZOOM
LENS = Material(kind=materials.GLASS, ior=1.5, transparency=1.0, reflectivity=0.0, specular=0.0, roughness=0.05,
                bevel_px=40.0, profile=materials.DOME, absorption=0.0)
FLAT = replace(LENS, bevel_px=2.0, profile=materials.BEVEL)  # flat faces: refracts only at the rim


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def cap_value():
    m = re.search(r"#define\s+CAUSTIC_CAP\s+([0-9.]+)", gl_glass._read("caustics_blur.frag"))
    return float(m.group(1)) if m else 4.0


def ellipse(cx, cy, rx, ry, n=96):
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def floor():
    return np.full((H, W, 3), 0.5)


def caustics(ctx, shapes, light_list, max_t=None, gain=None, period=None):
    """shared['caustics'] (h, w, 3 float) for shapes lit by light_list (lights.Light, world units)."""
    data = gh.build(shapes, floor(), light=False)
    # the harness's sdf is a bounding-circle lower bound away from the shapes (tiny steps there would run caustic
    # rays out of march steps): use the exact distance everywhere, as S's sdf gives
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    data["sdf"] = np.min([gh.poly_sdf(xs, ys, s.poly) for s in shapes], axis=0).astype(np.float32)
    world = SimpleNamespace(lights=light_list, ambient=(0.1, 0.1, 0.1))
    h = gh.Harness(ctx, data, world=world, period=period)
    saved = gl_glass.CAUSTIC_MAX_T, gl_glass.CAUSTIC_GAIN
    if max_t is not None:
        gl_glass.CAUSTIC_MAX_T = max_t
    if gain is not None:
        gl_glass.CAUSTIC_GAIN = gain
    caus, glass = gl_glass.make(ctx, gh.SIZE)
    caus.enabled = True
    try:
        frame = h.run([caus], h.frame())
        tex = frame.shared.get("caustics")
        out = h.read(tex)[..., :3].copy() if tex is not None else np.zeros((H, W, 3), np.float32)
    finally:
        gl_glass.CAUSTIC_MAX_T, gl_glass.CAUSTIC_GAIN = saved
        caus.release()
        glass.release()
    return out, data


def world_pos(x, y):
    return (x / ZOOM, y / ZOOM)


def test_directional(ctx):
    sun = lights.Light(kind=lights.DIRECTIONAL, dir=(0.0, 1.0), color=(1.0, 1.0, 1.0), intensity=1.0)
    c, _ = caustics(ctx, [gh.Shape(ellipse(450, 150, 120, 40), LENS, (0.9, 0.95, 1.0))], [sun])
    below = c[220:560, 330:570].max()
    above = c[0:100, :].mean()  # only the odd ray reflected back up off the lens tips
    check(below > 0.5, f"a. directional light: a lens focuses light below it (peak {below:.2f} > 0.5)")
    check(above < 0.01, f"a. directional light: next to nothing deposited above the lens (mean {above:.4f})")
    for ang in (30.0, 90.0, 180.0):
        a = np.radians(ang)
        lt = replace(sun, dir=(float(np.sin(a)), float(np.cos(a))))
        c2, _ = caustics(ctx, [gh.Shape(ellipse(450, 290, 120, 40), LENS, (0.9, 0.95, 1.0))], [lt])
        check(c2.max() > 0.05 and np.isfinite(c2).all(),
              f"a. directional light at {ang:.0f} deg: caustics present ({c2.max():.2f})")


def test_stack(ctx):
    spot = lights.Light(kind=lights.SPOT, pos=world_pos(450, 10), dir=(0.0, 1.0), color=(1.0, 1.0, 1.0),
                        intensity=1.0, radius=2000.0, falloff=1.0, cone_inner=10.0, cone_outer=16.0)
    single, _ = caustics(ctx, [gh.Shape(ellipse(450, 120, 110, 22), LENS, (0.9, 0.95, 1.0))], [spot])
    stack, _ = caustics(ctx, [gh.Shape(ellipse(450, 120 + 70 * i, 110, 22), LENS, (0.9, 0.95, 1.0))
                              for i in range(3)], [spot])
    p1 = single[300:584, 250:650].max()
    p3 = stack[300:584, 250:650].max()
    check(p3 > 0.25 * p1, f"b. 3-lens stack: light below the last lens {p3:.2f} > 25% of one lens's {p1:.2f}")


def baseline_common():
    try:
        return subprocess.run(["git", "-C", REPO, "show", f"{BASELINE_REV}:shaders/glass_common.glsl"],
                              capture_output=True, text=True, check=True).stdout
    except Exception:
        return None


def test_view_identical(ctx):
    base = baseline_common()
    if base is None:
        print(f"skip c. no git baseline {BASELINE_REV} for glass_common.glsl")
        return
    data = gh.build(gh.demo_shapes(), gh.checker(gh.SIZE))
    outs = []
    real_read = gl_glass._read
    rim = gl_glass.MARCH_RIM_STEPS  # (the baseline's resolve has no separate rim budget: compare at equal ones)
    gl_glass.MARCH_RIM_STEPS = gl_glass.MARCH_STEPS
    for src in (None, base):
        gl_glass._read = (real_read if src is None else
                          (lambda name, src=src: src if name == "glass_common.glsl" else real_read(name)))
        try:
            for sdf in (False, True):
                h = gh.Harness(ctx, data, sdf=sdf)
                caus, glass = gl_glass.make(ctx, gh.SIZE)
                h.run([caus, glass])
                outs.append(h.read().copy())
                caus.release()
                glass.release()
        finally:
            gl_glass._read = real_read
            gl_glass.MARCH_RIM_STEPS = rim
    for i, sdf in enumerate((False, True)):
        same = np.array_equal(outs[i], outs[i + 2])
        check(same, f"c. glass view output byte-identical to glass_common@{BASELINE_REV} (sdf {sdf})"
              + ("" if same else f": {(outs[i] != outs[i + 2]).any(axis=2).sum()} px differ"))


def test_cap_and_fade(ctx):
    cap = cap_value()
    pt = lights.Light(kind=lights.POINT, pos=world_pos(60, 290), color=(1.0, 1.0, 1.0), intensity=1.0,
                      radius=4000.0, falloff=1.0)
    c, _ = caustics(ctx, [gh.Shape(ellipse(250, 290, 40, 110), LENS, (0.9, 0.95, 1.0))], [pt], gain=30.0)
    check(c.max() <= cap * 1.05, f"d. a hard focus is soft-capped: max {c.max():.2f} <= CAUSTIC_CAP*1.05 "
                                 f"({cap * 1.05:.2f})")
    # a flat slab (no focusing) in a point light's beam, the path cut at max_t inside the view
    max_t = 500.0
    c, _ = caustics(ctx, [gh.Shape(rect(140, 150, 160, 430), FLAT, (0.9, 0.95, 1.0))], [pt], max_t=max_t)
    lum = c.max(axis=2) * 255.0
    worst, level = 0.0, 0.0
    for dy in range(-60, 61, 10):
        ang = np.arctan2(dy, 400.0)
        r = np.arange(0.6 * max_t, 1.1 * max_t)
        xs = np.round(60 + r * np.cos(ang)).astype(int)
        ys = np.round(290 + r * np.sin(ang)).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        prof = lum[ys[ok], xs[ok]]
        level = max(level, float(prof[:20].mean()))
        worst = max(worst, float(np.abs(np.diff(prof)).max()))
    check(level > 40.0, f"d. the beam behind a flat slab is lit before its end ({level:.0f}/255)")
    check(worst <= 10.0, f"d. the beam fades out at the path cap: largest step {worst:.1f}/255 <= 10")


def test_inside(ctx):
    pt = lights.Light(kind=lights.POINT, pos=world_pos(450, 290), color=(1.0, 1.0, 1.0), intensity=1.0,
                      radius=4000.0, falloff=1.0)
    disc = replace(LENS, bevel_px=2.0, profile=materials.BEVEL)
    c, _ = caustics(ctx, [gh.Shape(gh.ngon(450, 290, 60, 128), disc, (0.9, 0.95, 1.0))], [pt])
    lum = c.max(axis=2)
    a = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ring = lum[np.round(290 + 110 * np.sin(a)).astype(int), np.round(450 + 110 * np.cos(a)).astype(int)]
    mean = float(ring.mean())
    cv = float(ring.std() / max(mean, 1e-6))
    check(mean > 0.05, f"e. a light inside a glass disc lights the ring around it (mean {mean:.3f})")
    check(cv < 0.25, f"e. ... evenly, no starburst (variation {cv:.2f} < 0.25)")


def test_sliver(ctx):
    pt = lights.Light(kind=lights.POINT, pos=world_pos(100, 290), color=(1.0, 1.0, 1.0), intensity=1.0,
                      radius=4000.0, falloff=1.0)
    c, data = caustics(ctx, [gh.Shape(rect(300, 140, 301, 440), LENS, (0.9, 0.95, 1.0))], [pt])
    check((data["mat"][:, 300] > 0).sum() > 250 and (data["mat"][290, :] > 0).sum() <= 2,
          "f. the sliver is 1 px wide in the G-buffer")
    lum = c.max(axis=2)
    # behind the sliver, along a line parallel to it: what crossed it, unbent, falls off smoothly with the angle
    col = lum[200:380, 340]
    mean = float(col.mean())
    rough = float(np.abs(np.diff(col)).max() / max(mean, 1e-6))
    check(mean > 0.05, f"f. light crosses a 1 px glass sliver ({mean:.3f})")
    check(rough < 0.1, f"f. ... without streaks (largest step {rough:.3f} of the mean < 0.1)")


def test_periodic_directional(ctx):
    """A slanted sun through a flat glass band across the whole width of a periodic map: every ray crosses it once,
    so the light below it is even (no strip lit twice by a wrapped copy of a ray, none left dark)."""
    sun = lights.Light(kind=lights.DIRECTIONAL, dir=(0.3, 1.0), color=(1.0, 1.0, 1.0), intensity=1.0)
    for period in ((float(W), 0.0), (float(W), float(H))):
        c, _ = caustics(ctx, [gh.Shape(rect(-10, 100, W + 10, 130), FLAT, (0.9, 0.95, 1.0))], [sun], max_t=4000.0,
                        period=period)
        row = c[260, :, :].max(axis=1)
        mean = float(row.mean())
        cv = float(row.std() / max(mean, 1e-6))
        check(mean > 0.3 and cv < 0.1, f"h. periodic {period}: a sun through a glass band lights the floor below "
                                       f"evenly (mean {mean:.2f}, variation {cv:.3f} < 0.1)")
        if period[1] > 0:  # wrapping vertically too: above the band is below its copy one period up
            up = float(c[50, :, :].max(axis=1).mean())
            check(abs(up - mean) < 0.1 * mean, f"h. ... and above it, lit through the band's copy upstream "
                                                f"({up:.2f} ~ {mean:.2f})")


def test_garbage(ctx):
    nan, inf = float("nan"), float("inf")
    shape = [gh.Shape(ellipse(450, 290, 120, 40), LENS, (0.9, 0.95, 1.0))]
    good = lights.Light(kind=lights.POINT, pos=world_pos(450, 60), radius=2000.0, falloff=1.0)
    for label, lt in (("NaN pos", replace(good, pos=(nan, 0.0))), ("inf pos", replace(good, pos=(inf, 10.0))),
                      ("NaN dir spot", replace(good, kind=lights.SPOT, dir=(nan, 1.0))),
                      ("zero dir directional", replace(good, kind=lights.DIRECTIONAL, dir=(0.0, 0.0))),
                      ("NaN falloff", replace(good, falloff=nan)), ("inf dir directional",
                                                                    replace(good, kind=lights.DIRECTIONAL,
                                                                            dir=(inf, 1.0)))):
        try:
            c, _ = caustics(ctx, shape, [lt])
            ok = bool(np.isfinite(c).all())
            # lights.active (gl_glass's caustics, like gl_lights) drops a light without a finite position and repairs
            # the rest (dir -> (0, 1), falloff -> its default): those draw, as the light pass draws them
            if "pos" in label:
                ok = ok and c.max() < 1e-3
            check(ok, f"i. {label}: finite caustics, nothing drawn for a light without a finite position "
                  f"(max {np.nanmax(c):.3f})")
        except Exception as e:  # noqa: BLE001
            check(False, f"i. {label}: raised {e!r}")


def main():
    ctx = moderngl.create_standalone_context(require=410)
    test_directional(ctx)
    test_stack(ctx)
    test_view_identical(ctx)
    test_cap_and_fade(ctx)
    test_inside(ctx)
    test_sliver(ctx)
    test_periodic_directional(ctx)
    test_garbage(ctx)
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall caustics checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
