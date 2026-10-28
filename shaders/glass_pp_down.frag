// Push-pull inpainting, pull (down) pass: one level of a half-size pyramid of two fields, each rgb = mean over the
// valid pixels below, a = coverage (share of valid pixels). With u_first it makes the quarter-res level from every
// other full-res pixel in each direction (4 of each 4x4: what it inpaints is smooth or gets blurred anyway):
//   o_ratio  scene_lit / albedo (with RATIO_EPS) on background pixels (material id 0): the light reaching the floor
//   o_col    scene_lit on the background and opaque bodies: the surroundings glass / mirrors reflect
// gl_glass.py's resolve then knows, under a glass body, the light there (ratio) and a fallback colour.
// On a u_wrap axis (periodic, the view = one period) source texels past the level's edge wrap around to its start,
// so the pyramid is toroidal and has no seam there; otherwise they are skipped.
layout(location = 0) out vec4 o_ratio;
layout(location = 1) out vec4 o_col;

uniform bool u_first;
uniform sampler2D u_scene;
uniform sampler2D u_prev_ratio;
uniform sampler2D u_prev_col;
uniform bvec2 u_wrap;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy) * 2;
    vec4 sr = vec4(0.0), sc = vec4(0.0);
    ivec2 lim = u_first ? u_clip : textureSize(u_prev_ratio, 0);
    for (int k = 0; k < 4; k++) {
        ivec2 c = u_first ? 2 * o + 2 * ivec2(k & 1, k >> 1) : o + ivec2(k & 1, k >> 1);
        if (u_wrap.x) c.x %= lim.x;
        if (u_wrap.y) c.y %= lim.y;
        if (c.x >= lim.x || c.y >= lim.y) continue;
        if (u_first) {
            uint id = mat_id(c);
            vec3 s = texelFetch(u_scene, c, 0).rgb;
            if (id == 0u) sr += vec4((s + RATIO_EPS) / (texelFetch(u_albedo, c, 0).rgb + RATIO_EPS), 1.0);
            if (kind_of(id) <= KIND_OPAQUE) sc += vec4(s, 1.0);
        } else {
            vec4 r = texelFetch(u_prev_ratio, c, 0);
            vec4 q = texelFetch(u_prev_col, c, 0);
            sr += vec4(r.rgb * r.a, r.a);
            sc += vec4(q.rgb * q.a, q.a);
        }
    }
    o_ratio = sr.a > 0.0 ? vec4(sr.rgb / sr.a, 0.25 * sr.a) : vec4(1.0, 1.0, 1.0, 0.0);
    o_col = sc.a > 0.0 ? vec4(sc.rgb / sc.a, 0.25 * sc.a) : vec4(0.0);
}
