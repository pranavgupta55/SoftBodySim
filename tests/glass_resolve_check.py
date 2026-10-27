"""Assertions for the glass / mirror resolve (shaders/glass_resolve.frag) on the synthetic harness
(tests/glass_harness.py). Run from the repo root:

    .venv/bin/python3 tests/glass_resolve_check.py [out_dir]

a. clear glass isn't smoked: a glass disc of r = 50 px (transparency 1, absorption 0.01, casts_shadow False) keeps its
   centre within 15% of the lit floor there (absorption follows a capped thickness, not the body's size).
b. no comb on a mirror: on a flat, slightly rotated mirror rect under a bright bar, the reflection band's lower edge
   (where the reflected ray stops leaving the body) is a straight line: per column, its row deviates from a fitted
   line by a std of 1.5 px or less (the self walk-out is sub-pixel, not SDF-step quantised).
c. glints are normalised: roughness 1 with specular 1 adds a glint peak below 0.15 (on glass and on a mirror).
d. roughness blurs a mirror: the reflection of a striped bar in (b)'s band loses over half its contrast at
   roughness 1.
With out_dir, writes 4x crops of (b) at roughness 0 and 1.
Exit status 1 on a failure.
"""
import math
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

FAILS = []
GLASS_COL = (170 / 255, 215 / 255, 235 / 255)


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def run(ctx, data, **kw):
    h = gh.Harness(ctx, data, **kw)
    caus, glass = gl_glass.make(ctx, data["size"])
    h.run([caus, glass])
    out = h.read()[..., :3].astype(np.float64) / 255.0
    caus.release()
    glass.release()
    return out


def flat_floor(v):
    w, h = gh.SIZE
    return np.full((h, w, 3), v, np.float64)


def rot_rect(cx, cy, hw, hh, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return np.array([(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts])


def test_clear_glass(ctx):
    m = replace(materials.PRESETS["glass"], transparency=1.0, absorption=0.01, casts_shadow=False)
    cx, cy = 450, 300
    floor = flat_floor(0.6)
    data = gh.build([gh.Shape(gh.ngon(cx, cy, 50.0, 64), m, GLASS_COL)], floor)
    ref = gh.build([], floor)["scene"]  # the lit floor without the body
    for sdf in (False, True):
        out = run(ctx, data, sdf=sdf)
        c = out[cy - 2:cy + 3, cx - 2:cx + 3].reshape(-1, 3).mean(0)
        f = ref[cy - 2:cy + 3, cx - 2:cx + 3].reshape(-1, 3).mean(0)
        rel = float(np.max(np.abs(c - f) / f))
        check(rel <= 0.15, f"clear glass r=50: centre {np.round(c * 255).astype(int).tolist()} vs floor "
              f"{np.round(f * 255).astype(int).tolist()}: {rel:.0%} off (<= 15%) (sdf {sdf})")


BORDER_PX = 2.0
BAR_COL = (1.0, 0.95, 0.85)
BAR_LUM = sum(BAR_COL) / 3
STRIPE_PX = 12.0


def band_scene(rough, stripes=False):
    """A flat mirror rect rotated 7 degrees, a bright bar touching its (black, BORDER_PX) top edge, on a dark floor.
    With stripes the bar is cut into STRIPE_PX wide pieces, every other one dark."""
    mir = replace(materials.PRESETS["mirror"], roughness=rough, specular=0.0, reflectivity=1.0, bevel_px=1.0)
    bar = replace(materials.PRESETS["matte"], specular=0.0)
    deg, cx, cy, hw, hh = 7.0, 450.0, 330.0, 170.0, 90.0
    a = math.radians(deg)
    up = np.array([math.sin(a), -math.cos(a)])  # the rect's outward top normal
    along = np.array([math.cos(a), math.sin(a)])
    bc = np.array([cx, cy]) + up * (hh + BORDER_PX + 20.0)
    poly = rot_rect(cx, cy, hw, hh, deg)
    if stripes:
        n = int(2 * (hw + 10.0) // STRIPE_PX)
        bars = [gh.Shape(rot_rect(*(bc + along * ((k - (n - 1) / 2) * STRIPE_PX)), STRIPE_PX / 2, 20.0, deg), bar,
                         BAR_COL if k % 2 == 0 else (0.1, 0.12, 0.2)) for k in range(n)]
    else:
        bars = [gh.Shape(rot_rect(bc[0], bc[1], hw + 10.0, 20.0, deg), bar, BAR_COL)]
    data = gh.build(bars + [gh.Shape(poly, mir, (0.8, 0.8, 0.85))], flat_floor(0.15), light=False)
    add_border(data, poly, data["mats"].index(mir))
    return data, (cx, cy, hw, hh, a)


def add_border(data, poly, mid):
    """Like GLWorld's filled view: a BORDER_PX black outline around the body carrying its material id with a flat
    normal, part of the occluder; and an SDF quantised like the jump-flood one (whole-pixel distances)."""
    h, w = data["mat"].shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    sd = gh.poly_sdf(xs, ys, poly)
    ring = (sd >= 0) & (sd < BORDER_PX)
    data["mat"][ring] = mid
    data["normal"][ring] = 0.0
    data["dist"][ring] = 0.0
    data["albedo"][ring] = 0.0
    data["scene"][ring] = 0.0
    data["sdf"] = np.round(np.minimum(data["sdf"], sd - BORDER_PX)).astype(np.float32)


def top_row(geom, x):
    cx, cy, hw, hh, a = geom
    return cy - hh / math.cos(a) + (x - cx) * math.tan(a)  # the mirror's top edge at column x


def band_edges(out, geom):
    """Per column across the mirror's middle: the sub-pixel row where the band (bright, reflecting the bar) falls to
    half-way toward the interior's level."""
    cx, cy, hw, hh, a = geom
    lum = out.mean(-1)
    xs, rows = [], []
    for x in range(int(cx - 0.6 * hw), int(cx + 0.6 * hw)):
        top = top_row(geom, x)
        y0, y1 = int(math.ceil(top)) + 3, int(top) + 45
        col = lum[y0:y1, x]
        hi, lo = BAR_LUM, col[-5:].mean()  # the band shows the bar
        frac = (col - lo) / (hi - lo)
        below = np.nonzero(frac < 0.5)[0]
        if hi - lo < 0.1 or frac[0] < 0.8 or not len(below):
            continue  # no band at this column
        k = below[0]
        xs.append(x)
        rows.append(y0 + k - 1 + (frac[k - 1] - 0.5) / max(frac[k - 1] - frac[k], 1e-9))
    return np.array(xs, float), np.array(rows)


def band_contrast(out, geom):
    """Std of the luminance along the band, 4 px under the top edge (the stripes' reflection)."""
    cx, cy, hw, hh, a = geom
    lum = out.mean(-1)
    xs = np.arange(int(cx - 0.6 * hw), int(cx + 0.6 * hw))
    return float(np.std([lum[int(top_row(geom, x)) + 4, x] for x in xs]))


def test_mirror_band(ctx, out_dir):
    res = {}
    for rough in (0.0, 1.0):
        for stripes in (False, True):
            data, geom = band_scene(rough, stripes)
            res[rough, stripes] = run(ctx, data, sdf=True)
            if out_dir:
                cx, cy, hw, hh, _ = geom
                box = (int(cx - hw * 0.7), int(cy - hh - 40), int(cx + hw * 0.7), int(cy - hh + 50))
                gh.save_png(os.path.join(out_dir, f"resolve_band_rough{rough:g}{'_stripes' if stripes else ''}.png"),
                            gh.u8(gh.crop(res[rough, stripes], *box)), 4)
    xs, rows = band_edges(res[0.0, False], geom)
    ok = len(xs) > 100
    check(ok, f"mirror band: the bar's reflection found on {len(xs)} columns (> 100)")
    if ok:
        fit = np.polyval(np.polyfit(xs, rows, 1), xs)
        std = float(np.std(rows - fit))
        check(std <= 1.5, f"mirror band: edge row std about a line {std:.2f} px (<= 1.5, no comb)")
    c0, c1 = band_contrast(res[0.0, True], geom), band_contrast(res[1.0, True], geom)
    check(c0 > 0.15 and c1 < 0.5 * c0, f"mirror roughness blurs the reflection: stripe contrast {c0:.3f} at rough 0, "
          f"{c1:.3f} at rough 1 (< half)")


def test_glint(ctx):
    floor = flat_floor(0.4)
    for name, base in (("glass", replace(materials.PRESETS["glass"], bevel_px=40.0, profile=materials.DOME)),
                       ("mirror", replace(materials.PRESETS["mirror"], bevel_px=40.0, profile=materials.DOME))):
        outs = []
        for spec in (0.0, 1.0):
            m = replace(base, specular=spec, roughness=1.0)
            data = gh.build([gh.Shape(gh.ngon(450, 300, 60.0, 64), m, (0.6, 0.6, 0.65))], floor)
            outs.append(run(ctx, data, sdf=True))
        peak = float((outs[1] - outs[0]).max())
        check(peak < 0.15, f"glint: {name} roughness 1, specular 1 peak {peak:.3f} (< 0.15)")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=410)
    test_clear_glass(ctx)
    test_mirror_band(ctx, out_dir)
    test_glint(ctx)
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
