// Fluid field, splat (half resolution: each texel covers 2x2 render px): o_field = the share of those pixels a
// FLUID-kind body covers in the G-buffer (fill and outline); o_col = (albedo, 1) x share over the fluid's interior
// pixels (edge distance >= 1 px: not its outline), so after the blur rgb / a is the liquid's smoothed colour (the
// fluid preset colours every octagon differently).
layout(location = 0) out float o_field;
layout(location = 1) out vec4 o_col;

void main() {
    ivec2 h = ivec2(gl_FragCoord.xy);
    float f = 0.0;
    vec4 col = vec4(0.0);
    for (int k = 0; k < 4; k++) {
        ivec2 p = 2 * h + ivec2(k & 1, k >> 1);
        if (p.x >= u_clip.x || p.y >= u_clip.y || kind_of(mat_id(p)) != KIND_FLUID) continue;
        f += 0.25;
        if (edge_d(p) >= 1.0) col += 0.25 * vec4(texelFetch(u_albedo, p, 0).rgb, 1.0);
    }
    o_field = f;
    o_col = col;
}
