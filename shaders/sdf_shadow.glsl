// Soft shadows from the shared signed distance field (gl_sdf.py's shared["sdf"]: R32F, signed distance in render px
// to the nearest shadow-casting occluder boundary, negative inside, minimum-image across periodic edges; LINEAR
// filtered). Self-contained: include it into any shader (it declares no uniforms).
//
// Coordinates are render px, y down, continuous: texel (x, y) covers [x, x + 1) x [y, y + 1), so a fragment's
// own position is gl_FragCoord.xy. `period` is the map's repeat in render px per axis, (0, 0) = not periodic.
//
// sdf_shadow(sdf, p, dir, max_t, softness_px, period) -> 1 lit .. 0 shadowed, for a light at p + dir * max_t (dir
// a unit vector toward the light; for a directional light pass the march length you want, e.g. the screen
// diagonal). softness_px is the light's radius: the penumbra's angular half-width is softness_px / max_t, the
// classic k = max_t / softness_px of Inigo Quilez's min(k * d / t) soft shadow (iquilezles.org/articles/rmshadows,
// with his improved centred penumbra, so a penumbra extends both sides of the hard shadow edge).
// A start point inside an occluder first walks out of that body along dir (sphere tracing on |d|): a body never
// shadows itself (its faces are shaded by the lights' normals instead) but still shadows everything behind it.
// Bodies that touch merge in the field, so a pixel's walk out continues through a body pressed against it.
#ifndef SDF_SHADOW_GLSL
#define SDF_SHADOW_GLSL

const int SDF_EXIT_STEPS = 16;   // walk-out steps from inside a body
const int SDF_MARCH_STEPS = 64;   // sphere-tracing steps toward the light
const float SDF_MIN_STEP = 1.0;  // px
const float SDF_INSIDE_STEP = 0.5;  // px, march step while inside an occluder
const float SDF_OWN_MARGIN = 1.0;  // px: see sdf_shadow's own-body phase

// The field at p (bilinear). Off a non-periodic texture, a lower bound from the nearest texel centre (occluders
// all lie on the texture: sqrt(d^2 + offset^2) for outside points).
float sdf_at(sampler2D sdf, vec2 p, vec2 period) {
    vec2 size = vec2(textureSize(sdf, 0));
    if (period.x > 0.0) p.x -= period.x * floor(p.x / period.x);
    if (period.y > 0.0) p.y -= period.y * floor(p.y / period.y);
    vec2 c = clamp(p, vec2(0.5), size - 0.5);
    float d = texture(sdf, c / size).r;
    vec2 off = p - c;
    float o2 = dot(off, off);
    return (o2 > 0.0 && d > 0.0) ? sqrt(d * d + o2) : d;
}

float sdf_shadow(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period) {
    float t = 0.0;
    float d = sdf_at(sdf, p, period);
    for (int i = 0; i < SDF_EXIT_STEPS && d < 0.0; i++) {  // leave the body p is in
        t += max(-d, 0.5);
        if (t >= max_t) return 1.0;
        d = sdf_at(sdf, p + dir * t, period);
    }
    if (d < 0.0) return 1.0;  // still inside one long body: its own, so lit
    vec2 o = p + dir * t;
    float rem = max_t - t;
    float k = max_t / max(softness_px, 1e-3);
    float res = 1.0;
    float s = SDF_MIN_STEP;
    // After walking out, the ray first moves away from its own body (h rising, slowly when it left at a grazing
    // angle near the terminator): that stretch doesn't count, or the body would speckle-shadow its own lit face.
    // Counting starts once h falls SDF_OWN_MARGIN below the most it has been (something else ahead, or a concave
    // part of the same body); smaller dips are the rasterised outline's stairs, which a grazing ray re-enters.
    bool counting = t == 0.0;
    float hmax = d;
    for (int i = 0; i < SDF_MARCH_STEPS && s < rem; i++) {
        float h = sdf_at(sdf, o + dir * s, period);
        hmax = max(hmax, h);
        counting = counting || h < hmax - SDF_OWN_MARGIN;
        if (counting) {
            res = min(res, k * h / s);
            if (res <= -1.0) break;
        }
        // Inside an occluder (h < 0: a penumbra's dark side, the ray clipping a corner) step a fixed
        // SDF_INSIDE_STEP: stepping |h| there makes the sampled minimum of h jump as the step positions slide from
        // pixel to pixel, which drew wavy contour bands across penumbras. Deep inside, res hits -1 and breaks.
        s += h < 0.0 ? SDF_INSIDE_STEP : max(h, SDF_MIN_STEP);
    }
    res = clamp(res, -1.0, 1.0);
    return 0.25 * (1.0 + res) * (1.0 + res) * (2.0 - res);
}

#endif
