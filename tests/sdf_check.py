"""Checks for gl_sdf (offscreen, synthetic G-buffer via tests/sdf_harness.py). From the repo root:

    .venv/bin/python3 tests/sdf_check.py

- the SDF matches analytic signed distances to known polygons within 1 px (sign included off the boundary)
- shadows: a point behind an occluder is dark, open ground and a body's own pixels are lit, no terminator speckle
- periodic maps: minimum-image distances across the seam, and a shadow cast through the seam
- GI: off by default (no shared["gi"]); when on, light falls off from an emitter and an occluder blocks it
Exit status 1 on a failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdf_harness as H  # noqa: E402

import gl_sdf  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def analytic(polys, size, period=None):
    """Signed distance at every pixel centre to the union of polygons (with images one period over)."""
    w, h = size
    px, py = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    best = np.full((h, w), np.inf)
    for poly in polys:
        shifts = [(0.0, 0.0)]
        if period is not None:
            shifts = [(sx, sy) for sx in (0, -period[0], period[0]) for sy in (0, -period[1], period[1])]
        for sx, sy in shifts:
            d = H.poly_sdf(poly.pts + (sx, sy), px, py)
            best = np.where(np.abs(d) < np.abs(best), d, best)  # disjoint polygons: the nearest one decides
    return best


def sdf_vs_analytic(hz, polys, period=None, tag="", casting=None):
    """casting: the polygons that should be in the field (default: all)."""
    f = hz.run(hz.frame(polys, period=period))
    got = hz.sdf_array(f)
    want = analytic(polys if casting is None else casting, hz.size, period)
    err = np.abs(got - want)
    # vs the continuous polygon: sharp tips / corners lose sub-pixel slivers in rasterisation (up to 1 px there)
    check(err.max() <= 1.05, f"{tag}SDF within 1 px of analytic (max err {err.max():.3f}, mean {err.mean():.3f})")
    off = np.abs(want) > 1.0
    check(np.all(np.sign(got[off]) == np.sign(want[off])), f"{tag}SDF sign matches off the boundary")
    return f, got


def main():
    hz = H.Harness()
    w, h = hz.size
    check(not gl_sdf.GI_ENABLED, "GI disabled by default")

    # 1. SDF vs analytic: a square, a disc, a thin rotated bar (the shadow-casting ones)
    polys = H.scene_default()
    f, sdf = sdf_vs_analytic(hz, polys[:4])
    check("gi" not in f.shared, "no shared['gi'] while GI is off")
    # the glow polygon (no casts-shadow flag) is not in the field
    f2, sdf2 = sdf_vs_analytic(hz, polys, tag="with a non-casting glow body: ", casting=polys[:4])
    check(sdf2[200, 180] > 100, "non-shadow-casting body absent from the SDF")

    # 2. shadows: light above the square (430, 300, half-diagonal 45)
    light = (430.0, 150.0)
    sq = [H.Poly(H.regular(430, 300, 45, 4, np.pi / 4), (0.8, 0.3, 0.3))]
    fs = hz.run(hz.frame(sq))
    for soft in (2.0, 16.0):
        s = hz.shadow(fs, light, soft)
        check(s[420, 430] < 0.05, f"soft {soft}: point behind the occluder is shadowed ({s[420, 430]:.3f})")
        check(s[300, 600] > 0.95, f"soft {soft}: open ground beside it is lit ({s[300, 600]:.3f})")
        check(s[275, 430] > 0.95 and s[325, 430] > 0.95,
              f"soft {soft}: the body's own near and far faces are not self-shadowed "
              f"({s[275, 430]:.3f}, {s[325, 430]:.3f})")
        check(s[150:200, 380:480].min() > 0.95, f"soft {soft}: between light and body all lit")
    s2, s16 = hz.shadow(fs, light, 2.0), hz.shadow(fs, light, 16.0)
    row = 500  # across the shadow edge below the square: the softer light has the wider penumbra
    pen2 = np.sum((s2[row] > 0.05) & (s2[row] < 0.95))
    pen16 = np.sum((s16[row] > 0.05) & (s16[row] < 0.95))
    check(pen16 > pen2 >= 1, f"penumbra widens with softness ({pen2} px -> {pen16} px)")
    # no terminator speckle on a round body: every pixel inside the disc is lit by a light beside it
    disc = [H.Poly(H.regular(680, 420, 50, 32), (0.8, 0.7, 0.3))]
    fd = hz.run(hz.frame(disc))
    sd = hz.shadow(fd, (560.0, 330.0), 2.0)
    inside = hz.sdf_array(fd) < 0
    check(sd[inside].min() > 0.95, f"disc interior fully lit, no speckle (min {sd[inside].min():.3f})")

    # 3. periodic: a box across the left / right seam, and one near the left edge only (wrap needs the JFA)
    per = (float(w), float(h))
    seam = [H.Poly(np.array([[-40, 250], [40, 250], [40, 330], [-40, 330]], float), (0.8, 0.3, 0.6)),
            H.Poly(np.array([[10, 450], [50, 450], [50, 500], [10, 500]], float), (0.3, 0.7, 0.7))]
    fp, sp = sdf_vs_analytic(hz, seam, period=per, tag="periodic: ")
    check(abs(sp[475, 890] - 19.5) <= 1.0, f"periodic: min-image distance across the seam ({sp[475, 890]:.2f})")
    # light just right of the seam box; a point at x=150 lies behind the box through the seam
    s = hz.shadow(fp, (800.0, 290.0), 4.0)
    check(s[290, 150] < 0.05, f"periodic: shadow cast through the seam ({s[290, 150]:.3f})")
    check(s[30, 150] > 0.95, f"periodic: off the shadow, lit ({s[30, 150]:.3f})")
    fn = hz.run(hz.frame(seam[1:]))  # (reuses the effect's texture: after the periodic checks)
    check(hz.sdf_array(fn)[475, 890] > 800, "non-periodic: no wrap")

    # 4. GI: an emitter with an occluder to its left
    gl_sdf.GI_ENABLED = True
    try:
        scene = [H.Poly(H.regular(450, 300, 25, 6), (1.0, 0.6, 0.2), flags=H.OCCLUDER | H.LIT, mat=2),
                 H.Poly(np.array([[360, 220], [375, 220], [375, 380], [360, 380]], float), (0.8, 0.8, 0.8))]
        fg = hz.run(hz.frame(scene))
        check("gi" in fg.shared, "GI on publishes shared['gi']")
        g = hz.gi_array(fg)[..., :3].sum(-1)
        near, far = g[300, 520], g[300, 800]
        check(near > far > 0, f"GI falls off with distance ({near:.4f} -> {far:.4f})")
        blocked, open_ = g[300, 300], g[300, 600]
        check(blocked < 0.5 * open_, f"occluder blocks GI ({blocked:.4f} behind vs {open_:.4f} in the open)")
        check(np.isfinite(g).all(), "GI finite")
    finally:
        gl_sdf.GI_ENABLED = False

    for e in hz.effects:
        e.release()
    print(f"{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
