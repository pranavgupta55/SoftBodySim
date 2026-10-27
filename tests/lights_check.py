"""Checks for gl_lights (lights + compose) on the synthetic G-buffer of tests/lit_harness.py. Run from the repo root:

    .venv/bin/python3 tests/lights_check.py

Exit status 1 on the first failed check.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lit_harness as lh  # noqa: E402
import gl_lights  # noqa: E402
from lights import Light  # noqa: E402

AMB = (0.25, 0.25, 0.3)
FLOOR_Y = 545  # the floor box's middle row: flat, matte
CYAN_BOX = (90, 120)  # lit flag clear
BG = (450, 20)
GLOSSY = (450, 260)  # glossy circle centre
TOON = (700, 300)
GLOW = (820, 120)


def lum(img, x, y):
    return float(img[y, x].astype(np.float32).sum())


def ambient_only(h, x, y):
    a = h.scene["albedo"][y, x, :3].astype(np.float32) / 255.0
    return np.round(np.clip(a * np.array(AMB), 0, 1) * 255.0)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def main():
    h = lh.Harness()
    albedo = h.scene["albedo"][..., :3]

    img = h.run([], ambient=AMB)
    for x, y in ((300, FLOOR_Y), (200, 300), (GLOSSY)):
        check(np.abs(img[y, x] - ambient_only(h, x, y)).max() <= 1, f"no lights: ({x}, {y}) = albedo x ambient")
    check((img[CYAN_BOX[1], CYAN_BOX[0]] == albedo[CYAN_BOX[1], CYAN_BOX[0]]).all(), "unlit flag: albedo unchanged")
    check((img[BG[1], BG[0]] == albedo[BG[1], BG[0]]).all(), "background passes through")
    # (albedo x ambient + white emissive is over 1 in every channel: the tone shoulder rolls it off just under white)
    check(img[GLOW[1], GLOW[0]].min() >= 230, "glow material: emissive added")
    gl_lights.TONEMAP = False
    img = h.run([], ambient=AMB)
    gl_lights.TONEMAP = True
    check((img[GLOW[1], GLOW[0]] == 255).all(), "TONEMAP off: emissive clips to white")

    # point light: nearer is brighter, past the radius is ambient only, unlit stays albedo even under the light
    lt = Light(pos=(200, 480), radius=300, intensity=1.5, height=40)
    img = h.run([lt], ambient=AMB, sdf=False)
    check(lum(img, 220, FLOOR_Y) > lum(img, 320, FLOOR_Y) > lum(img, 420, FLOOR_Y), "point: nearer pixels brighter")
    check(np.abs(img[FLOOR_Y, 700] - ambient_only(h, 700, FLOOR_Y)).max() <= 1, "point: nothing past its radius")
    img = h.run([Light(pos=CYAN_BOX, radius=300, intensity=3.0)], ambient=AMB)
    check((img[CYAN_BOX[1], CYAN_BOX[0]] == albedo[CYAN_BOX[1], CYAN_BOX[0]]).all(), "unlit flag under a light")
    check((h.read("light")[CYAN_BOX[1], CYAN_BOX[0], :3] == 0).all(), "unlit flag: no light accumulated")
    check((h.read("light")[BG[1], BG[0], :3] == 0).all(), "background: no light accumulated")

    # bevels: the floor's top edge faces up, so a light above lights it more than the flat middle at the same reach
    img = h.run([Light(pos=(450, 300), radius=600, height=40)], ambient=(0, 0, 0), sdf=False)
    check(lum(img, 450, 517) > lum(img, 450, 530), "bevel facing the light is brighter than the flat top")

    # spot: inside the cone lit, outside (same distance) dark
    spot = Light(kind="spot", pos=(450, 400), dir=(0.0, 1.0), radius=400, cone_inner=10, cone_outer=15, height=40)
    img = h.run([spot], ambient=AMB, sdf=False)
    check(lum(img, 450, FLOOR_Y) > lum(h.run([], ambient=AMB), 450, FLOOR_Y) + 20, "spot: inside the cone is lit")
    img = h.run([spot], ambient=AMB, sdf=False)
    check(np.abs(img[FLOOR_Y, 600] - ambient_only(h, 600, FLOOR_Y)).max() <= 1, "spot: outside the cone is dark")
    check(np.abs(img[FLOOR_Y, 300] - ambient_only(h, 300, FLOOR_Y)).max() <= 1, "spot: other side too")

    # spot quads: the cone's box covers the whole cone, whatever the direction and width
    lit = ((h.scene["normal"][..., 3] & 4) > 0) & (h.scene["mat"] != lh.MAT_ID["toon"])  # toon rounds dim to 0
    ys, xs = np.mgrid[0:h.size[1], 0:h.size[0]] + 0.5
    for d, outer in (((1.0, 0.0), 30), ((-0.6, -0.8), 50), ((0.0, -1.0), 100), ((0.7, 0.7), 5)):
        sp = Light(kind="spot", pos=(450, 330), dir=d, radius=350, cone_inner=outer * 0.3, cone_outer=outer,
                   falloff=0.0, height=1e4)
        h.run([sp], sdf=False)
        got = h.read("light")[..., 0] > 0
        vx, vy = xs - 450, ys - 330
        dist = np.hypot(vx, vy)
        cosang = (vx * d[0] + vy * d[1]) / np.maximum(dist, 1e-6) / np.hypot(*d)
        want = lit & (dist < 349) & (cosang > np.cos(np.radians(outer)) + 0.01)
        check(not (want & ~got).any(), f"spot box covers its cone (dir {d}, {outer} deg)")
        check(not (got & ~((dist < 351) & (cosang > np.cos(np.radians(outer)) - 0.01))).any(), "spot: nothing outside")

    # directional: flat tops equally lit everywhere, the facing bevel brighter than the far one
    sun = Light(kind="directional", dir=(1.0, 0.0), intensity=0.8)
    img = h.run([sun], ambient=AMB, sdf=False)
    check(abs(lum(img, 100, FLOOR_Y) - lum(img, 800, FLOOR_Y)) <= 3, "directional: uniform on flat surfaces")
    check(lum(img, 22, FLOOR_Y) > lum(img, 877, FLOOR_Y), "directional: the bevel facing it is brighter")

    # shadows: the floor behind the yellow box (330, 430) from a light above it, with and without the SDF
    lt = Light(pos=(330, 300), radius=600, height=40)
    with_sdf, no_sdf = h.run([lt], ambient=AMB), h.run([lt], ambient=AMB, sdf=False)
    check(lum(with_sdf, 330, FLOOR_Y) < lum(no_sdf, 330, FLOOR_Y) - 20, "sdf: occluded pixel is shadowed")
    check(abs(lum(with_sdf, 600, 300) - lum(no_sdf, 600, 300)) <= 3, "sdf: open pixel is not")
    off = h.run([Light(pos=(330, 300), radius=600, height=40, casts_shadows=False)], ambient=AMB)
    check((off == no_sdf).all(), "casts_shadows=False skips the shadow")

    # specular: glossy gets a highlight, matte none
    h.run([Light(pos=(450, 200), radius=500, height=80)], ambient=AMB, sdf=False)
    spec = h.read("spec")
    check(spec[200:260, 420:480, 0].max() > 0.2, "glossy: specular highlight")
    check(spec[FLOOR_Y, :, :3].max() == 0, "matte: no specular")

    # toon bands: diffuse takes only bands + 1 values
    h.run([Light(pos=(620, 200), radius=500, color=(1, 1, 1), height=60)], ambient=AMB, sdf=False)
    light = h.read("light")
    m = h.scene["mat"] == lh.MAT_ID["toon"]
    vals = np.unique(np.round(light[..., 0][m], 3))
    check(len(vals) <= 4 and set(np.round(vals * 3, 2)) <= {0.0, 1.0, 2.0, 3.0}, f"toon: band levels {vals}")

    # periodic: a light just right of the left seam lights the right edge too (minimum image), not without a period
    lt = Light(pos=(5, FLOOR_Y), radius=120, height=40)
    h.run([lt], ambient=AMB, sdf=False, period=(900.0, 0.0))
    check(h.read("light")[FLOOR_Y, 870, 0] > 0.05, "periodic: lights across the seam")
    h.run([lt], ambient=AMB, sdf=False)
    check(h.read("light")[FLOOR_Y, 870, 0] == 0, "no period: no wrap")

    # zoom: world units x zoom = px (position, radius, height, softness)
    a = h.run([Light(pos=(300, 450), radius=300, height=40, softness=8)], ambient=AMB)
    b = h.run([Light(pos=(150, 225), radius=150, height=20, softness=4)], ambient=AMB, zoom=2.0)
    check(np.abs(a.astype(int) - b).max() <= 1, "zoom scales light positions and sizes")

    # many lights: 300 (two batches) dim lights equal one bright one
    one = Light(pos=(450, 450), radius=400, intensity=1.5, color=(1, 1, 1))
    # past F4_MIN_LIGHTS lights the targets are RGBA32F (RGBA16F would round each of the 300 additions to half an
    # ulp, 2^-12 below 1, drifting ~8% at a sum near 1)
    gl_lights.LIGHT_DTYPE = "f4"
    h.run([one], sdf=False)
    a = h.read("light")[FLOOR_Y, 650, 0]
    gl_lights.LIGHT_DTYPE = None
    h.run([one], sdf=False)
    check(h.frame.shared["light"].dtype == "f2", "1 light: RGBA16F targets")
    h.run([one.copy(intensity=0.005)] * 300, sdf=False)
    check(h.frame.shared["light"].dtype == "f4", "300 lights: RGBA32F targets")
    b = h.read("light")[FLOOR_Y, 650, 0]
    check(abs(a - b) < 0.01 * a, f"300 lights add up like one (fp32 sums: {a:.4f} vs {b:.4f})")
    h.run([one.copy(intensity=0.0)] * 299 + [one], sdf=False)
    check(abs(h.read("light")[FLOOR_Y, 650, 0] - a) < 1e-6, "the 300th light (second batch) draws")
    h.run([one] * 30, sdf=False)
    check(h.frame.shared["light"].dtype == "f4", "30 lights after 300: still RGBA32F (hysteresis)")
    h.run([one] * 24, sdf=False)
    check(h.frame.shared["light"].dtype == "f2", "24 lights: back to RGBA16F")
    h.run([one.copy(enabled=False)], sdf=False)
    check(h.read("light").max() == 0, "disabled lights draw nothing")

    # graceful degradation: compose alone (no light / spec / gi / caustics) = albedo x ambient + emissive
    img = h.run([], ambient=AMB, sdf=False, only=(300,))
    check(np.abs(img[FLOOR_Y, 300] - ambient_only(h, 300, FLOOR_Y)).max() <= 1, "compose without the lights pass")
    gi = h.ctx.texture(h.size, 4, np.full((h.size[1], h.size[0], 4), 0.5, np.float16).tobytes(), dtype="f2")
    img = h.run([], ambient=AMB, sdf=False, shared={"gi": gi})
    a = h.scene["albedo"][FLOOR_Y, 300, :3] / 255.0
    check(np.abs(img[FLOOR_Y, 300] - np.round(a * (np.array(AMB) + 0.5) * 255)).max() <= 1, "compose adds gi")
    gi.release()
    check("scene_lit" in h.frame.shared and h.frame.shared["scene_lit"].read() == h.gw.tex.read(), "scene_lit copy")

    # toon light quantisation in compose (LIGHT_LEVELS) changes the image, off by default
    base = h.run([Light(pos=(450, 450), radius=400)], ambient=AMB, sdf=False)
    gl_lights.LIGHT_LEVELS = 4
    q = h.run([Light(pos=(450, 450), radius=400)], ambient=AMB, sdf=False)
    gl_lights.LIGHT_LEVELS = 0
    check((q != base).any() and len(np.unique(q[FLOOR_Y, 300:600, 0])) < len(np.unique(base[FLOOR_Y, 300:600, 0])),
          "LIGHT_LEVELS quantises")

    # tone mapping (TONEMAP, on by default): mid tones unchanged (only the dither, < 1/255), over-bright colours keep
    # their hue
    lt = Light(pos=(450, 450), radius=400, intensity=1.0)
    toned = h.run([lt], ambient=AMB, sdf=False).astype(int)
    gl_lights.TONEMAP = False
    plain = h.run([lt], ambient=AMB, sdf=False).astype(int)
    gl_lights.TONEMAP = True
    mid = plain.max(axis=2) <= 0.8 * 255
    check(np.abs(toned - plain)[mid].max() <= 1, "tone map: pixels below the knee unchanged within 1/255")
    flat = h.read("light")[FLOOR_Y, :, 0]
    x5 = int(np.argmin(np.abs(flat - 0.5)))
    check(abs(flat[x5] - 0.5) < 0.05 and np.abs(toned[FLOOR_Y, x5] - plain[FLOOR_Y, x5]).max() <= 1,
          f"a pixel with light 0.5 unchanged within 1/255 (x={x5})")
    white = lambda im: int((im >= 254).all(axis=2).sum())  # noqa: E731
    check(white(toned) <= white(plain), f"tone map: no more flat white ({white(toned)} vs {white(plain)} px)")
    # an orange emitter at x3 strength: the glow material's row swapped for one
    table = np.frombuffer(h.material_tex.read(), np.float32).reshape(-1, lh.MAT_TEXELS * 4).copy()
    saved_table = table.copy()
    orange = lh.materials.Material(emissive=(1.0, 0.5, 0.1), emissive_strength=3.0, casts_shadow=False)
    table[lh.MAT_ID["glow"]] = lh.pack_material(orange)
    h.material_tex.write(table.tobytes())
    img = h.run([], ambient=AMB, sdf=False).astype(float)
    a = h.scene["albedo"][GLOW[1], GLOW[0], :3] / 255.0
    want = a * np.array(AMB) + np.array(orange.emissive) * 3.0
    got = img[GLOW[1], GLOW[0]]
    ratio_want, ratio_got = want[1] / want[0], got[1] / got[0]
    check(abs(ratio_got / ratio_want - 1) < 0.05, f"orange emissive x3 keeps its g/r ({ratio_got:.3f} vs "
                                                  f"{ratio_want:.3f}) and stays orange {got}")
    h.material_tex.write(saved_table.tobytes())
    # world.exposure scales the lit colour; bad values count as 1
    h.run([lt], ambient=AMB, sdf=False)
    check(gl_lights.exposure(SimpleNamespace(exposure=float("nan"))) == 1.0
          and gl_lights.exposure(SimpleNamespace()) == 1.0 and gl_lights.exposure(SimpleNamespace(exposure=0)) == 0.0,
          "exposure: default 1, NaN -> 1, 0 kept")

    # the G-buffer's albedo may be GLWorld.tex itself: compose reads a copy
    lt = Light(pos=(450, 450), radius=400)
    ref = h.run([lt], ambient=AMB, sdf=False)
    h.gw.tex.write(np.ascontiguousarray(h.scene["albedo"]).tobytes())
    saved = h.gbuf["albedo"]
    h.gbuf["albedo"] = h.gw.tex
    alias = h.run([lt], ambient=AMB, sdf=False)
    h.gbuf["albedo"] = saved
    check((alias == ref).all(), "albedo aliasing GLWorld.tex")
    h.release()
    gi_checks()
    print("all lights checks passed")


def gi_checks():
    """GI (gl_sdf) from an emitter whose material is unlit: none (rc_scene gates emission by Mat.lit); lit: some.
    And a world with no lights and no ambient keeps the dim default ambient (user decision: no full-bright default)."""
    import lights
    import gl_sdf
    import sdf_harness as sh
    hz = sh.Harness()
    table = np.frombuffer(hz.mats.read(), np.float32).reshape(hz.n_mats, -1).copy()
    glow = lh.materials.PRESETS["glow"]
    gl_sdf.GI_ENABLED = True
    try:
        res = {}
        for lit in (True, False):
            table[2] = np.asarray(lh.materials.pack(lh.materials.Material(**{**glow.__dict__, "lit": lit})),
                                  np.float32).ravel()
            hz.mats.write(table.tobytes())
            flags = sh.OCCLUDER | (sh.LIT if lit else 0)
            scene = [sh.Poly(sh.regular(450, 300, 25, 6), (1.0, 0.6, 0.2), flags=flags, mat=2)]
            res[lit] = float(hz.gi_array(hz.run(hz.frame(scene)))[..., :3].max())
        check(res[True] > 0.01, f"a lit emitter gives GI ({res[True]:.3f})")
        check(res[False] == 0.0, f"an unlit emitter gives zero GI ({res[False]})")
    finally:
        gl_sdf.GI_ENABLED = False
        for e in hz.effects:
            e.release()
    check(lights.of_world(SimpleNamespace())[1] == lights.DEFAULT_AMBIENT
          and lights.of_world(SimpleNamespace(lights=[], ambient=None))[1] == lights.DEFAULT_AMBIENT,
          "no lights, no ambient: the default (dim) ambient")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL", e)
        sys.exit(1)
