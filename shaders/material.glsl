// The material table (gl_world.GLWorld.material_tex): RGBA32F, MAT_TEXELS texels per material, one row per material
// id (0 = no material). Texel layout: materials.py (pack). Include via gl_world's _load / FrameInfo.load includes.
// Constants mirror materials.py.
const int MAT_TEXELS = 8;
const int KIND_OPAQUE = 0, KIND_GLASS = 1, KIND_MIRROR = 2, KIND_FLUID = 3;
const int PROFILE_BEVEL = 0, PROFILE_DOME = 1, PROFILE_PILLOW = 2;
const int PATTERN_NONE = 0, PATTERN_NOISE = 1, PATTERN_STRIPES = 2, PATTERN_CHECKER = 3, PATTERN_DOTS = 4;
// G-buffer normal.a flag bits (x 255)
const int FLAG_OCCLUDER = 1, FLAG_CASTS_SHADOW = 2, FLAG_LIT = 4;
// a body's black border (filled view): lit, but it doesn't glow with the body nor turn see-through / mirror
const int FLAG_BORDER = 8;

struct Mat {
    vec3 tint;
    float tint_strength;
    vec3 emissive;
    float emissive_strength;
    float specular;
    float roughness;
    int bands;
    float bevel_px;
    int profile;
    int kind;
    float ior;
    float reflectivity;
    float transparency;
    float absorption;
    int pattern;
    float pattern_scale;
    float pattern_strength;
    bool casts_shadow;
    bool lit;
};

Mat mat_fetch(sampler2D table, uint id) {
    int row = int(id);
    vec4 t0 = texelFetch(table, ivec2(0, row), 0);
    vec4 t1 = texelFetch(table, ivec2(1, row), 0);
    vec4 t2 = texelFetch(table, ivec2(2, row), 0);
    vec4 t3 = texelFetch(table, ivec2(3, row), 0);
    vec4 t4 = texelFetch(table, ivec2(4, row), 0);
    vec4 t5 = texelFetch(table, ivec2(5, row), 0);
    Mat m;
    m.tint = t0.rgb;
    m.tint_strength = t0.a;
    m.emissive = t1.rgb;
    m.emissive_strength = t1.a;
    m.specular = t2.x;
    m.roughness = t2.y;
    m.bands = int(t2.z + 0.5);
    m.bevel_px = t2.w;
    m.profile = int(t3.x + 0.5);
    m.kind = int(t3.y + 0.5);
    m.ior = t3.z;
    m.reflectivity = t3.w;
    m.transparency = t4.x;
    m.absorption = t4.y;
    m.pattern = int(t4.z + 0.5);
    m.pattern_scale = t4.w;
    m.pattern_strength = t5.x;
    m.casts_shadow = t5.y > 0.5;
    m.lit = t5.z > 0.5;
    return m;
}

// G-buffer flags (normal.a x 255) of a body pixel with this material: an occluder, casts a shadow, is lit.
int mat_flags(Mat m) {
    return FLAG_OCCLUDER | (m.casts_shadow ? FLAG_CASTS_SHADOW : 0) | (m.lit ? FLAG_LIT : 0);
}

// The body's own colour with the material's tint mixed in (tint_strength 0: unchanged).
vec3 mat_tinted(Mat m, vec3 col) {
    return mix(col, col * m.tint, m.tint_strength);
}

// Slope h'(d) of the surface height profile at distance d (px) inside the nearest edge; 0 past bevel_px.
// BEVEL: a 45 degree chamfer; DOME: a quarter circle (steep at the rim, capped); PILLOW: a soft quadratic roll-off.
float mat_slope(Mat m, float d) {
    float b = m.bevel_px;
    if (b <= 0.0 || d >= b) {
        return 0.0;
    }
    float t = clamp(d / b, 0.0, 1.0);
    if (m.profile == PROFILE_DOME) {
        float u = 1.0 - t;
        return min(u / sqrt(max(1.0 - u * u, 1e-4)), 4.0);
    } else if (m.profile == PROFILE_PILLOW) {
        return 2.0 * (1.0 - t);
    }
    return 1.0;
}

float mat_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float mat_value_noise(vec2 q) {
    vec2 i = floor(q), f = fract(q);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mat_hash(i), mat_hash(i + vec2(1.0, 0.0)), u.x),
               mix(mat_hash(i + vec2(0.0, 1.0)), mat_hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

// The material's procedural pattern at q (pattern cells: rest-frame render px / pattern_scale): 1 = untouched albedo,
// 0 = darkest. The albedo is scaled by 1 - pattern_strength * (1 - value) (mat_patterned).
float mat_pattern_value(int pattern, vec2 q) {
    if (pattern == PATTERN_NOISE) {
        return 0.65 * mat_value_noise(q) + 0.35 * mat_value_noise(q * 2.03 + 17.0);
    } else if (pattern == PATTERN_STRIPES) {
        return step(0.5, fract(q.x));
    } else if (pattern == PATTERN_CHECKER) {
        return mod(floor(q.x) + floor(q.y), 2.0);
    } else if (pattern == PATTERN_DOTS) {
        return step(0.3, length(fract(q) - 0.5));
    }
    return 1.0;
}

// The pattern's average value over many cells (what a sub-pixel pattern should read as): dots cover pi 0.3^2 of
// a cell, noise averages about 0.5.
float mat_pattern_mean(int pattern) {
    return pattern == PATTERN_DOTS ? 1.0 - 3.14159265 * 0.09 : 0.5;
}

// col with the pattern applied at rest-frame position rest_px (render px). Cells under about 2 px can't be drawn
// (they alias into moire), so the pattern fades to its mean there; big cells are untouched.
vec3 mat_patterned(Mat m, vec3 col, vec2 rest_px) {
    if (m.pattern == PATTERN_NONE || m.pattern_strength <= 0.0) {
        return col;
    }
    float cell_px = max(m.pattern_scale, 1e-3);
    float v = mat_pattern_value(m.pattern, rest_px / cell_px);
    v = mix(mat_pattern_mean(m.pattern), v, smoothstep(1.0, 2.5, cell_px));
    return col * (1.0 - m.pattern_strength * (1.0 - v));
}
