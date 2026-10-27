// Shared by the world layer's shaders (gl_world.py prepends it after the #version line).
//
// The world framebuffer is the render surface's size and is addressed like a pygame surface: pixel (x, y), y down,
// lands on texel (x, y), so the composite reads it exactly like the uploaded pygame layers (row 0 = the top row).
// Clip space is set up so integer pixel corners map to texel corners: the pixel a fragment belongs to is
// ivec2(gl_FragCoord.xy). Draw order between bodies comes from a depth layer per primitive (later = nearer).

uniform vec2 u_size;   // framebuffer size, px
uniform ivec2 u_clip;  // pixels at or past this (the world view's size) are clipped, like a pygame subsurface

// Lighting on (gl_world draws into the G-buffer): world programs write all three outputs, f_color (location 0: the
// albedo, the unlit colour plus material tint / pattern), f_normal (1) and f_mat (2) (research/lighting_contract.md).
// Lighting off: the same programs draw into GLWorld.fbo, which has only location 0; f_color is then exactly the unlit
// colour and the other outputs go nowhere.
uniform bool u_lit;

// G-buffer normal texel: rg = n.xy * 0.5 + 0.5 (+y down), b = edge distance px / 255, a = flags / 255
vec4 gbuf_normal(vec2 n, float d, int flags) {
    return vec4(n * 0.5 + 0.5, clamp(d / 255.0, 0.0, 1.0), float(flags) / 255.0);
}

const float LAYERS = 1048576.0;  // 2^20 layers: exact in float32 and in a 24-bit depth buffer

vec4 to_clip(vec2 px, int layer) {
    return vec4(px / u_size * 2.0 - 1.0, 1.0 - 2.0 * float(layer + 1) / LAYERS, 1.0);
}

vec3 unpack_rgb(int c) {
    return vec3(float(c & 255), float((c >> 8) & 255), float((c >> 16) & 255)) / 255.0;
}

#ifdef VERTEX_SHADER
// The quad corner for gl_VertexID 0..3 of a 4-vertex triangle strip over the pixel box [lo, hi) (hi exclusive).
vec2 box_corner(ivec2 lo, ivec2 hi) {
    return vec2((gl_VertexID & 1) == 1 ? hi.x : lo.x, (gl_VertexID & 2) == 2 ? hi.y : lo.y);
}
#endif

// floor(a / b) and ceil(a / b) for b > 0 (GLSL's / on negative ints is implementation defined)
int floordiv(int a, int b) {
    return a >= 0 ? a / b : -((b - 1 - a) / b);
}

int ceildiv(int a, int b) {
    return -floordiv(-a, b);
}
