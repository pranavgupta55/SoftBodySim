"""Assertions for materials.pack's sanitiser, of_body's glow hue and material.glsl's sub-pixel pattern fade. Run
from the repo root:

    .venv/bin/python3 tests/materials_pack_check.py

1. every preset (and a spread of in-range materials) packs byte-identically to the plain, unsanitised packing.
2. NaN, +-inf, negative, out-of-range and non-number values of every field pack to finite values inside
   materials.RANGES / ENUMS (NaN -> the field's default), and pack never raises.
3. resolve / material_over keep the raw values (only pack sanitises).
4. of_body: a glowing body whose emissive colour nobody set glows in its fill colour's hue; an explicit emissive,
   a non-glowing body, a black body and a preset with its own emissive colour are left alone.
5. material.glsl mat_patterned: a sub-pixel pattern (pattern_scale 0.5) is one flat value (its mean, no moire);
   big cells (scale >= 2.5) are exactly the raw pattern.
Exit status 1 on a failure.
"""
import os
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import materials as M  # noqa: E402
from materials import Material  # noqa: E402

FAILS = []
NAN, INF = float("nan"), float("inf")


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def raw_pack(m):
    """The packing before the sanitiser (materials.pack's layout, values as given)."""
    rows = [(*m.tint, m.tint_strength), (*m.emissive, m.emissive_strength),
            (m.specular, m.roughness, m.bands, m.bevel_px), (m.profile, m.kind, m.ior, m.reflectivity),
            (m.transparency, m.absorption, m.pattern, m.pattern_scale),
            (m.pattern_strength, float(m.casts_shadow), float(m.lit), 0.0)]
    out = np.zeros((M.TEXELS, 4), np.float32)
    out[:len(rows)] = rows
    return out


def in_range(packed):
    """Problems of one packed row against RANGES / ENUMS (empty: fine)."""
    u = M.unpack(packed)
    bad = [] if np.isfinite(packed).all() else ["non-finite"]
    for name, (lo, hi) in M.RANGES.items():
        vals = getattr(u, name)
        for v in (vals if isinstance(vals, tuple) else (vals,)):
            if not lo <= v <= hi:
                bad.append(f"{name}={v}")
    for name, allowed in M.ENUMS.items():
        if getattr(u, name) not in allowed:
            bad.append(f"{name}={getattr(u, name)}")
    return bad


def check_identity():
    mats = dict(M.PRESETS)
    mats["edges"] = Material(tint=(0.0, 1.0, 0.5), tint_strength=1.0, emissive=(0.0, 0.0, 0.0), emissive_strength=8.0,
                             specular=1.0, roughness=0.0, bands=16, bevel_px=64.0, profile=M.PILLOW, kind=M.FLUID,
                             ior=3.0, reflectivity=1.0, transparency=1.0, absorption=0.0, pattern=M.PATTERN_DOTS,
                             pattern_scale=0.5, pattern_strength=1.0, casts_shadow=False, lit=False)
    mats["mid"] = Material(tint=(0.3, 0.7, 2.5), tint_strength=0.37, emissive=(0.1, 0.2, 0.9), emissive_strength=1.7,
                           roughness=0.123, bands=3, bevel_px=4.5, ior=1.33, absorption=0.031, pattern_scale=6.25)
    for name, m in mats.items():
        check(M.pack(m).tobytes() == raw_pack(m).tobytes(), f"in-range '{name}' packs byte-identically")


def check_sanitise():
    defaults = Material()
    bad_values = [NAN, INF, -INF, -1.0, -1e9, 1e9, 2.5, "x", None]
    for f in fields(Material):
        if f.name in ("casts_shadow", "lit"):
            continue
        for v in bad_values:
            val = (v, v, v) if isinstance(f.default, tuple) else v
            try:
                packed = M.pack(replace(defaults, **{f.name: val}))
            except Exception as e:  # noqa: BLE001
                check(False, f"pack {f.name}={v!r} raised {type(e).__name__}: {e}")
                continue
            bad = in_range(packed)
            if bad:
                check(False, f"pack {f.name}={v!r} -> finite and in range ({bad})")
    check(True, "every field x NaN / inf / negative / huge / non-number packs finite and in range")
    for f in fields(Material):
        if f.name in ("casts_shadow", "lit"):
            continue
        val = (NAN,) * 3 if isinstance(f.default, tuple) else NAN
        got = M.unpack(M.pack(replace(defaults, **{f.name: val})))
        check(getattr(got, f.name) == field_default(f.name), f"NaN {f.name} packs as its default")
    u = M.unpack(M.pack(replace(defaults, roughness=3.0, specular=5.0, ior=100.0, emissive_strength=50.0,
                                bevel_px=1e6, bands=1000, pattern_scale=0.01, kind=7, pattern=-2, profile=1.6)))
    check((u.roughness, u.specular, u.ior, u.emissive_strength, u.bevel_px, u.bands, u.pattern_scale)
          == (1.0, 1.0, 3.0, 8.0, 64.0, 16, 0.5), "out-of-range values clamp to their bounds")
    check((u.kind, u.pattern, u.profile) == (M.OPAQUE, M.PATTERN_NONE, M.PILLOW),
          "unknown enums fall back to the default; near-integers round like mat_fetch")
    u = M.unpack(M.pack(replace(defaults, tint=(-1.0, INF, 0.5), emissive=(NAN, -3.0, 1.0))))
    check(u.tint == (0.0, 16.0, 0.5) and u.emissive == (1.0, 0.0, 1.0), "colours: >= 0, finite, NaN -> default")


def field_default(name):
    """A field's default as it reads back from the float32 table."""
    v = getattr(Material(), name)
    return tuple(float(np.float32(c)) for c in v) if isinstance(v, tuple) else type(v)(np.float32(v))


def check_resolve_raw():
    m = M.resolve("glossy", {"roughness": 3.0, "ior": NAN})
    check(m.roughness == 3.0 and m.ior != m.ior, "resolve keeps raw override values (pack alone sanitises)")


def check_glow_hue():
    def body(material, over=None, fill=(220, 40, 40), **kw):
        return SimpleNamespace(material=material, material_over=over, fill_col=fill, **kw)
    m = M.of_body(body("glow"))
    check(np.allclose(m.emissive, (1.0, 40 / 220, 40 / 220), atol=1e-3), f"bare glow body glows its hue {m.emissive}")
    check(M.of_body(body("glow", fill=(0, 0, 0))).emissive == (1.0, 1.0, 1.0), "black glow body keeps white")
    check(M.of_body(body("glow", {"emissive": (0.2, 0.9, 1.0)})).emissive == (0.2, 0.9, 1.0),
          "an explicit emissive wins")
    check(M.of_body(body("matte")) == M.resolve("matte"), "non-glowing body: material unchanged (shared id)")
    m = M.of_body(body("matte", {"emissive_strength": 0.5}, fill=[40, 200, 40]))
    check(np.allclose(m.emissive, (0.2, 1.0, 0.2)), "a glow slider on a matte body glows its hue")
    m = M.of_body(body("glow", wall_side=0))
    check(not m.casts_shadow and np.allclose(m.emissive, (1.0, 40 / 220, 40 / 220), atol=1e-3),
          "walls keep casts_shadow off")


PATTERN_FRAG = """
#version 330
uniform sampler2D u_table;
uniform float u_scale;
uniform int u_pattern;
uniform int u_raw;
out vec4 frag;
void main() {
    Mat m = mat_fetch(u_table, 1u);
    m.pattern = u_pattern;
    m.pattern_scale = u_scale;
    m.pattern_strength = 1.0;
    vec2 p = gl_FragCoord.xy * 0.731 + vec2(3.17, 1.9);  // (an arbitrary rest frame, not pixel aligned)
    float v = u_raw == 1 ? mat_pattern_value(u_pattern, p / u_scale) : mat_patterned(m, vec3(1.0), p).r;
    frag = vec4(v, 0.0, 0.0, 1.0);
}
"""


def check_pattern_fade():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context(require=410)
    except Exception as e:  # noqa: BLE001
        check(False, f"no GL context for the pattern check: {e}")
        return
    with open(os.path.join(REPO, "shaders", "material.glsl")) as fh:
        lib = fh.read()
    src = PATTERN_FRAG.replace("out vec4 frag;", lib + "\nout vec4 frag;")
    prog = ctx.program(vertex_shader="#version 330\nin vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }",
                       fragment_shader=src)
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f", "p")])
    table = ctx.texture((M.TEXELS, 2), 4, np.stack([M.pack(Material())] * 2).tobytes(), dtype="f4")
    table.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex = ctx.texture((96, 96), 4, dtype="f4")
    fbo = ctx.framebuffer([tex])

    def draw(pattern, scale, raw=0):
        fbo.use()
        table.use(0)
        prog["u_table"].value, prog["u_pattern"].value, prog["u_scale"].value = 0, pattern, scale
        prog["u_raw"].value = raw
        vao.render(moderngl.TRIANGLE_STRIP)
        return np.frombuffer(tex.read(), np.float32).reshape(96, 96, 4)[:, :, 0]

    for pat, name in ((M.PATTERN_STRIPES, "stripes"), (M.PATTERN_CHECKER, "checker"), (M.PATTERN_DOTS, "dots"),
                      (M.PATTERN_NOISE, "noise")):
        sub = draw(pat, 0.5)
        check(float(sub.max() - sub.min()) < 1e-5, f"{name} at scale 0.5: flat (spread {sub.max() - sub.min():.4f})")
        raw = draw(pat, 0.5, raw=1)
        check(abs(float(sub.mean()) - float(raw.mean())) < 0.12,
              f"{name} at scale 0.5: flat value {sub.mean():.3f} ~ the raw pattern's mean {raw.mean():.3f}")
        for scale in (2.5, 12.0, 400.0):
            check(np.array_equal(draw(pat, scale), draw(pat, scale, raw=1)), f"{name} at scale {scale}: untouched")
    for r in (vao, vbo, prog, table, fbo, tex):
        r.release()
    ctx.release()


def main():
    check_identity()
    check_sanitise()
    check_resolve_raw()
    check_glow_hue()
    check_pattern_fade()
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall materials pack checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
