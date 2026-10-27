// Jump flooding, one step pass (gl_sdf.py): each texel keeps the nearest seed among its own and those of the 8
// texels u_step away (minimum-image distance on periodic axes, lookups wrap there). Run with steps
// 2^(n-1) .. 1 and then 1 again (JFA+1). Needs sdf_common.glsl.
uniform isampler2D u_seeds;
uniform int u_step;
uniform vec2 u_period;  // texels, (0, 0) none

out ivec2 f_seed;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_seeds, 0);
    vec2 pc = vec2(p);
    ivec2 best = ivec2(SDF_NONE);
    float best_d = 1e30;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 q = p + ivec2(dx, dy) * u_step;
            if (!sdf_wrap(q, size, u_period)) continue;
            ivec2 s = texelFetch(u_seeds, q, 0).xy;
            if (s.x == SDF_NONE) continue;
            vec2 d = sdf_min_image(vec2(s) - pc, u_period);
            float l = dot(d, d);
            if (l < best_d) {
                best_d = l;
                best = s;
            }
        }
    }
    f_seed = best;
}
