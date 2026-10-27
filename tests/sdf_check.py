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


# The march variants on their own (sdf_shadow.glsl): u_mode 0 sdf_shadow, 1 sdf_shadow_pt (point light at u_light),
# 2 sdf_shadow_dir (light along u_dir, march u_reach, fade from u_fade).
VARIANT_FRAG = """
uniform sampler2D u_sdf;
uniform int u_mode;
uniform vec2 u_light;
uniform vec2 u_dir;
uniform float u_reach;
uniform float u_fade;
uniform float u_soft;
uniform vec2 u_per;
out vec4 f_color;
void main() {
    vec2 p = gl_FragCoord.xy;
    float sh = 1.0;
    if (u_mode == 2) {
        sh = sdf_shadow_dir(u_sdf, p, u_dir, u_reach, u_soft, u_per, u_fade);
    } else {
        vec2 dl = u_light - p;
        if (u_per.x > 0.0) dl.x -= u_per.x * floor(dl.x / u_per.x + 0.5);
        if (u_per.y > 0.0) dl.y -= u_per.y * floor(dl.y / u_per.y + 0.5);
        float L = length(dl);
        if (L > 0.5) {
            sh = u_mode == 1 ? sdf_shadow_pt(u_sdf, p, dl / L, L, u_soft, u_per)
                             : sdf_shadow(u_sdf, p, dl / L, L, u_soft, u_per);
        }
    }
    f_color = vec4(sh);
}
"""
H._EXTRA["sdf_check_variants.frag"] = VARIANT_FRAG
_variant_prog = None


def variant(hz, f, mode, soft, light=(0.0, 0.0), direction=(1.0, 0.0), reach=0.0, fade=0.0):
    """(h, w) shadow factor of one sdf_shadow.glsl variant over the frame."""
    global _variant_prog
    if _variant_prog is None:
        _variant_prog = f.load("world_base.vert", "sdf_check_variants.frag", ["sdf_shadow.glsl"])
    p = _variant_prog
    f.shared["sdf"].use(0)
    for name, v in (("u_sdf", 0), ("u_mode", mode), ("u_light", tuple(map(float, light))),
                    ("u_dir", tuple(map(float, direction))), ("u_reach", float(reach)), ("u_fade", float(fade)),
                    ("u_soft", float(soft)), ("u_per", gl_sdf._period(f.period))):
        if name in p:
            p[name].value = v
    hz.shadow_fbo.use()
    f.fullscreen(p)
    return np.frombuffer(hz.shadow_out.read(), np.float32).reshape(hz.size[1], hz.size[0], 4)[..., 0]


def bar(x0, y0, x1, y1):
    return H.Poly(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), (0.7, 0.7, 0.7))


def march_checks(hz):
    w, h = hz.size
    # (a) a light inside an occluder lights the open ground around it, also on the body's far side
    fa = hz.run(hz.frame([bar(380, 260, 480, 340)]))
    for soft in (2.0, 8.0):
        s = variant(hz, fa, 1, soft, light=(430.0, 300.0))
        far = (s[300, 650], s[500, 430], s[100, 200])
        check(min(far) > 0.5, f"light inside an occluder (soft {soft}) lights the open ground around it "
                              f"({', '.join(f'{v:.2f}' for v in far)})")
        old = variant(hz, fa, 0, soft, light=(430.0, 300.0))
        check(old[300, 650] < 0.05, f"(plain sdf_shadow still treats it as blocked: {old[300, 650]:.2f})")
    # a light on a body's surface still lets the body shadow its far side, and lights its own side
    s = variant(hz, fa, 1, 4.0, light=(480.5, 300.0))
    check(s[300, 200] < 0.05 and s[300, 700] > 0.9 and s[150, 600] > 0.9,
          f"light on a surface: far side dark ({s[300, 200]:.2f}), its own side lit "
          f"({s[300, 700]:.2f}, {s[150, 600]:.2f})")

    # (b) a directional light parallel to a wall: nothing lit between the walls that block it
    walls = [bar(0, 0, w, 8), bar(w - 8, 0, w, h), bar(0, h - 8, w, h), bar(0, 0, 8, h)]
    fb = hz.run(hz.frame(walls))
    for name, direction, region in (("dir 0", (1.0, 0.0), (slice(8, h - 8), slice(8, w - 30))),
                                    ("dir 90", (0.0, 1.0), (slice(8, h - 30), slice(8, w - 8)))):
        s = variant(hz, fb, 2, 4.0, direction=direction, reach=float(np.hypot(w, h)))
        check(s[region].max() < 0.1, f"{name}: no lit sliver along a parallel wall (max {s[region].max():.3f})")

    # (c) a 1 px rope casts a continuous shadow (its raster stairs used to leave lit stripes)
    a, b = np.array([300.0, 100.0]), np.array([340.0, 400.0])
    n = np.array([-(b - a)[1], (b - a)[0]])
    n /= np.linalg.norm(n)
    rope = H.Poly(np.array([a + n / 2, b + n / 2, b - n / 2, a - n / 2]), (0.8, 0.8, 0.8))
    fc = hz.run(hz.frame([rope]))
    for soft in (0.0, 1.0):
        row = hz.shadow(fc, (380.0, 60.0), soft)[450, 150:300]
        check(row.std() < 0.03 and row.max() < 0.1,
              f"1 px rope, soft {soft}: continuous shadow (std {row.std():.3f}, max {row.max():.3f})")

    # (d) periodic directional shadow fades out along the march instead of ending in a cut
    per = (float(w), float(h))
    fd = hz.run(hz.frame([bar(400, 250, 470, 330)], period=per))
    reach = 1.5 * w
    dvec = np.array([0.8, 0.6])
    # along the shadow's axis (it wraps around the map), from just past the box to beyond the march length
    d = np.arange(80.0, reach + 100.0)
    pts = np.array([435.0, 290.0]) - d[:, None] * dvec
    xs = np.floor(pts[:, 0] % w).astype(int)
    ys = np.floor(pts[:, 1] % h).astype(int)
    steps, ends = {}, {}
    for fade in (0.0, 0.5 * reach):
        s = variant(hz, fd, 2, 4.0, direction=dvec, reach=reach, fade=fade)
        prof = s[ys, xs]
        steps[fade] = float(np.abs(np.diff(prof)).max()) * 255
        ends[fade] = (float(prof[:200].max()), float(prof[-50:].min()))
    check(steps[0.5 * reach] <= 10.0 and ends[0.5 * reach][0] < 0.05 and ends[0.5 * reach][1] > 0.95,
          f"periodic directional shadow fades out: no neighbour step > 10/255 along it (max "
          f"{steps[0.5 * reach]:.1f}, dark {ends[0.5 * reach][0]:.2f} by the box, lit {ends[0.5 * reach][1]:.2f} "
          f"past the march; without the fade the step is {steps[0.0]:.1f})")


def spec_check():
    """(e) roughness 1 + specular 1: a broad sheen, not a white flood (light_accum's lobe normalisation)."""
    import lit_harness as lh
    from lights import Light
    hl = lh.Harness()
    rows = np.frombuffer(hl.material_tex.read(), np.float32).reshape(-1, lh.MAT_TEXELS * 4).copy()
    m = lh.materials.Material(specular=1.0, roughness=1.0)
    rows[lh.MAT_ID["glossy"]] = lh.pack_material(m)
    hl.material_tex.write(rows.tobytes())
    hl.run([Light(pos=(450, 150), intensity=1.0, radius=900, falloff=1.0, height=120, softness=4)], only=(200,))
    spec = hl.read("spec")[180:340, 370:530, :3].max()
    check(spec < 0.15, f"roughness 1, specular 1: spec per light < 0.15 (max {spec:.3f})")
    rows[lh.MAT_ID["glossy"]] = lh.pack_material(lh.materials.Material(specular=1.0, roughness=0.2))
    hl.material_tex.write(rows.tobytes())
    hl.run([Light(pos=(450, 150), intensity=1.0, radius=900, falloff=1.0, height=120, softness=4)], only=(200,))
    tight = hl.read("spec")[180:340, 370:530, :3].max()
    check(tight > 0.5, f"roughness 0.2: the tight highlight keeps its strength (max {tight:.3f})")
    hl.release()


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

    march_checks(hz)
    spec_check()

    for e in hz.effects:
        e.release()
    print(f"{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
