// Helpers shared by the jump flooding passes (jfa_seed / jfa_step / sdf_dist) and the radiance cascades (rc_*),
// gl_sdf.py includes it. Texel (x, y) is render pixel (x, y), row 0 = top, as in gl_world.
//
// Periodic maps: `period` is the repeat length in texels per axis (0 = that axis does not repeat). Wrapping assumes
// the period starts at texel 0; a period shorter than the texture wraps lookups into [0, period), a longer one
// leaves the texels past the texture's end unknown (treated as missing).
#ifndef SDF_COMMON_GLSL
#define SDF_COMMON_GLSL

const int SDF_NONE = -32768;  // "no seed" in the RG16I seed textures

// q wrapped onto the texture along periodic axes; false if it is off the texture.
bool sdf_wrap(inout ivec2 q, ivec2 size, vec2 period) {
    ivec2 L = ivec2(period + 0.5);
    if (L.x > 0) q.x -= L.x * int(floor(float(q.x) / float(L.x)));
    if (L.y > 0) q.y -= L.y * int(floor(float(q.y) / float(L.y)));
    return all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, size));
}

// The minimum-image version of offset d on a torus of `period` (axes with period 0 are left alone).
vec2 sdf_min_image(vec2 d, vec2 period) {
    if (period.x > 0.0) d.x -= period.x * floor(d.x / period.x + 0.5);
    if (period.y > 0.0) d.y -= period.y * floor(d.y / period.y + 0.5);
    return d;
}

// Is texel q of the source solid? mode 0: a G-buffer `normal` texture, flags in alpha (bit0 occluder, bit1 casts
// shadow, both needed); mode 1: any texture, solid where alpha > 0.5.
bool sdf_solid(sampler2D src, ivec2 q, int mode) {
    vec4 t = texelFetch(src, q, 0);
    if (mode == 0) {
        uint f = uint(t.a * 255.0 + 0.5);
        return (f & 3u) == 3u;
    }
    return t.a > 0.5;
}

#endif
