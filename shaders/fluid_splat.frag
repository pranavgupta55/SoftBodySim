// Fluid field, splat (half resolution: each texel covers 2x2 render px): o_field = the share of those pixels a
// FLUID-kind body covers in the G-buffer (fill and outline); o_col = (albedo, 1) x share over the fluid's interior
// pixels (edge distance >= 1 px: not its outline), so after the blur rgb / a is the liquid's smoothed colour (the
// fluid preset colours every octagon differently); o_ratio the same for the light ratio (scene_lit - spec + eps) /
// (albedo + eps) there: the light the fluid bodies themselves receive (a spot over the pool), which the resolve
// lights the liquid with. L's per-pixel spec is left out of it: it follows each body's own dome (speckle).
layout(location = 0) out float o_field;
layout(location = 1) out vec4 o_col;
layout(location = 2) out vec4 o_ratio;

uniform sampler2D u_scene;
uniform sampler2D u_spec;
uniform bool u_spec_on;

void main() {
    ivec2 h = ivec2(gl_FragCoord.xy);
    float f = 0.0;
    vec4 col = vec4(0.0), ratio = vec4(0.0);
    for (int k = 0; k < 4; k++) {
        ivec2 p = 2 * h + ivec2(k & 1, k >> 1);
        if (p.x >= u_clip.x || p.y >= u_clip.y || kind_of(mat_id(p)) != KIND_FLUID) continue;
        f += 0.25;
        if (edge_d(p) >= 1.0) {
            vec3 a = texelFetch(u_albedo, p, 0).rgb;
            vec3 s = texelFetch(u_scene, p, 0).rgb;
            if (u_spec_on) s = max(s - texelFetch(u_spec, p, 0).rgb, 0.0);
            col += 0.25 * vec4(a, 1.0);
            ratio += 0.25 * vec4((s + RATIO_EPS) / (a + RATIO_EPS), 1.0);
        }
    }
    o_field = f;
    o_col = col;
    o_ratio = ratio;
}
