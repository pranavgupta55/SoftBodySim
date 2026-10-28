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
e. no seam on a periodic map: a mirror straddling x = 0 with the period = the view; the column jump across the seam
   is at most 2x the median jump between neighbouring columns of the mirror (the push-pull fill wraps).
f. what lies under glass shows through: an opaque red rect under a glass disc, with F's behind layer, shows red
   (its red within 20% of the rect's own lit red), lit directly (the stand-in's shared["light"]) and by the
   push-pull ratio.
With out_dir, writes 4x crops of (b) at roughness 0 and 1, the seam and the rect under glass.
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


def stand_in_light(data):
    """What L's shared["light"] would hold for glass_harness.build's stand-in lighting: its bevel-shaded point light
    (build's formula) without the ambient, as an (h, w, 3) array."""
    h, w = data["mat"].shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    lx, ly = gh.LIGHT_POS
    vx, vy, vz = lx - xs, ly - ys, np.full_like(xs, 80.0)
    vl = np.sqrt(vx * vx + vy * vy + vz * vz)
    n2 = data["normal"]
    nz = np.sqrt(np.maximum(1 - (n2 ** 2).sum(-1), 0))
    ndl = np.maximum((n2[..., 0] * vx + n2[..., 1] * vy + nz * vz) / vl, 0)
    fall = np.clip(1 - np.hypot(vx, vy) / gh.LIGHT_RADIUS, 0, 1) ** 2
    return np.repeat((0.9 * fall * ndl)[..., None], 3, axis=-1)


def run_behind(ctx, data, under, light=True, **kw):
    """Like run(), with F's behind layer (gbuf behind / behind_nrm / behind_mat) taken from `under` (the same
    build without the see-through bodies) and, with `light`, the stand-in's shared["light"]: the resolve's direct
    path (glass_behind.frag)."""
    h = gh.Harness(ctx, data, **kw)
    normal = np.zeros(under["normal"].shape[:2] + (4,), np.uint8)
    normal[..., :2] = gh.u8(under["normal"] * 0.5 + 0.5)
    normal[..., 2] = np.clip(np.round(under["dist"]), 0, 255).astype(np.uint8)
    normal[..., 3] = np.where(under["mat"] > 0, 7, 4)  # occluder / casts shadow / lit; background lit
    size = data["size"]
    h.gbuf["behind"] = h._tex(gh.rgba8(under["albedo"]), size, 4, "f1")
    h.gbuf["behind_nrm"] = h._tex(normal, size, 4, "f1")
    h.gbuf["behind_mat"] = h._tex(under["mat"].astype(np.uint16), size, 1, "u2")
    frame = h.frame()
    if light:
        frame.shared["light"] = h._tex(stand_in_light(data).astype(np.float32), size, 3, "f4")
    caus, glass = gl_glass.make(ctx, size)
    h.run([caus, glass], frame)
    out = h.read()[..., :3].astype(np.float64) / 255.0
    caus.release()
    glass.release()
    return out


def test_body_under_glass(ctx, out_dir):
    """(e) An opaque red rect under a glass disc shows red through the glass, close to its own lit colour."""
    red = gh.Shape(gh.rounded_rect(430, 300, 60, 80, 4), replace(materials.PRESETS["matte"], specular=0.0),
                   (220 / 255, 60 / 255, 60 / 255))
    glass = gh.Shape(gh.ngon(470, 300, 75.0, 64), materials.PRESETS["glass"], GLASS_COL)
    floor = flat_floor(0.5)
    data, under = gh.build([red, glass], floor), gh.build([red], floor)
    for light in (True, False):
        out = run_behind(ctx, data, under, light=light)
        # glass interior pixels over the rect (away from both rims)
        ys, xs = np.mgrid[0:data["mat"].shape[0], 0:data["mat"].shape[1]] + 0.5
        sel = (np.hypot(xs - 470, ys - 300) < 60) & (np.abs(xs - 430) < 50) & (np.abs(ys - 300) < 70)
        c, ref = out[sel].mean(0), under["scene"][sel].mean(0)
        rel = float(abs(c[0] - ref[0]) / ref[0])  # (green / blue: + the glass's own colour, 1 - transparency of it)
        check(c[0] > 1.6 * c[1] and c[0] > 1.6 * c[2] and rel <= 0.2,
              f"red rect under glass shows red: {np.round(c * 255).astype(int).tolist()} vs its lit colour "
              f"{np.round(ref * 255).astype(int).tolist()} (red {rel:.0%} off, <= 20%) (shared light {light})")
        if out_dir and light:
            gh.save_png(os.path.join(out_dir, "resolve_red_under_glass.png"),
                        gh.u8(gh.crop(out, 340, 200, 570, 400)), 2)


def test_periodic_seam(ctx, out_dir):
    """(d) A mirror straddling x = 0 on a periodic map (period = the view): the column jump across the seam is at most
    2x the median jump between neighbouring columns in the mirror (the push-pull fill is toroidal)."""
    w, h = gh.SIZE
    mir = replace(materials.PRESETS["mirror"], specular=0.0, roughness=0.3)
    shapes = [gh.Shape(gh.ngon(x, 300, 60.0, 64), mir, (0.8, 0.8, 0.85)) for x in (0.0, float(w))]
    shapes += [gh.Shape(gh.rounded_rect(120, 200, 30, 20, 4), materials.PRESETS["matte"], (0.9, 0.3, 0.2)),
               gh.Shape(gh.rounded_rect(780, 420, 30, 20, 4), materials.PRESETS["matte"], (0.2, 0.4, 0.9))]
    # a smooth floor and light, both periodic over the view's width (no edge of their own at the seam; the stand-in
    # light isn't periodic): the light differs a lot either side of the mirror (0.69 at x = 60, 0.41 at x = 840)
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    wave = 0.12 * np.sin(2 * np.pi * 3 * xs / w) * np.cos(2 * np.pi * ys / 160) + 0.05 * np.cos(2 * np.pi * xs / w)
    floor = np.stack([0.5 + wave, 0.5 + 0.8 * wave, 0.55 + 0.5 * wave], -1)
    data = gh.build(shapes, floor, light=False)
    data["scene"] = data["albedo"] * (0.55 + 0.35 * np.sin(2 * np.pi * xs / w))[..., None]
    res = {}
    for period in ((float(w), 0.0), None):
        out = run(ctx, data, sdf=False, period=period)
        lum = out.mean(-1)
        rows = slice(300 - 40, 300 + 40)
        seam = float(np.abs(lum[rows, 0] - lum[rows, w - 1]).mean())
        cols = list(range(0, 50)) + list(range(w - 51, w - 1))  # inside the mirror, either side of the seam
        jumps = [float(np.abs(lum[rows, x + 1] - lum[rows, x]).mean()) for x in cols if x != w - 1]
        res[period] = (seam, float(np.median(jumps)))
        if out_dir and period:
            both = np.concatenate([out[200:400, w - 100:], out[200:400, :100]], 1)
            gh.save_png(os.path.join(out_dir, "resolve_seam_mirror.png"), gh.u8(both), 3)
    seam, med = res[(float(w), 0.0)]
    seam0, med0 = res[None]
    # (a jump under one 8-bit step is rounding: the floor of the bound)
    check(seam <= max(2.0 * med, 1.0 / 255), f"periodic seam through a mirror: column jump {seam:.4f} at the seam vs "
          f"median {med:.4f} (<= 2x, or 1/255); without the period {seam0:.4f} vs {med0:.4f}")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=410)
    test_clear_glass(ctx)
    test_mirror_band(ctx, out_dir)
    test_glint(ctx)
    test_periodic_seam(ctx, out_dir)
    test_body_under_glass(ctx, out_dir)
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
