// Two jump floods at once, one step pass (gl_sdf.py): jfa_step.frag on both seed pairs of an RGBA16I texture (xy
// and zw are independent floods sharing the texel fetches). Needs sdf_common.glsl.
uniform isampler2D u_seeds;
uniform int u_step;
uniform vec2 u_period;  // texels, (0, 0) none

out ivec4 f_seed;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_seeds, 0);
    vec2 pc = vec2(p);
    ivec4 best = ivec4(SDF_NONE);
    vec2 best_d = vec2(1e30);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 q = p + ivec2(dx, dy) * u_step;
            if (!sdf_wrap(q, size, u_period)) continue;
            ivec4 s = texelFetch(u_seeds, q, 0);
            if (s.x != SDF_NONE) {
                vec2 d = sdf_min_image(vec2(s.xy) - pc, u_period);
                float l = dot(d, d);
                if (l < best_d.x) {
                    best_d.x = l;
                    best.xy = s.xy;
                }
            }
            if (s.z != SDF_NONE) {
                vec2 d = sdf_min_image(vec2(s.zw) - pc, u_period);
                float l = dot(d, d);
                if (l < best_d.y) {
                    best_d.y = l;
                    best.zw = s.zw;
                }
            }
        }
    }
    f_seed = best;
}
