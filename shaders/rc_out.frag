// Radiance cascades, output pass (gl_sdf.py GiEffect): cascade 0 (per-texel fluence on the half-res grid E)
// upsampled bilinearly to the render size as shared["gi"]. Solid texels (bodies) have no light inside them in the
// cascades, so a body px takes the GI just outside its nearest edge (stepping out along the distance field's
// gradient): a body is lit on the side facing an emitter. Needs sdf_common.glsl.
uniform sampler2D u_c0;     // E, cascade 0 (LINEAR)
uniform sampler2D u_dist;   // E, signed distance in texels (NEAREST)
uniform vec2 u_period;      // texels of E, (0, 0) none
uniform float u_gain;

out vec4 f_gi;

float dist_at(ivec2 q, ivec2 size) {
    if (!sdf_wrap(q, size, u_period)) return 1e4;
    return texelFetch(u_dist, q, 0).r;
}

void main() {
    ivec2 size = textureSize(u_c0, 0);
    vec2 pe = gl_FragCoord.xy * 0.5;  // render px -> E texels
    ivec2 q = ivec2(floor(pe));
    float d = dist_at(q, size);
    if (d < 0.0) {
        vec2 grad = vec2(dist_at(q + ivec2(1, 0), size) - dist_at(q - ivec2(1, 0), size),
                         dist_at(q + ivec2(0, 1), size) - dist_at(q - ivec2(0, 1), size));
        if (dot(grad, grad) > 0.0) pe += normalize(grad) * (-d + 2.0);
    }
    // wrapped into [0.5, period + 0.5): the bilinear lookup always has both neighbours (the grid runs past the
    // period, GiEffect._alloc), so there is no seam at the wrap
    if (u_period.x > 0.0) pe.x -= u_period.x * floor((pe.x - 0.5) / u_period.x);
    if (u_period.y > 0.0) pe.y -= u_period.y * floor((pe.y - 0.5) / u_period.y);
    f_gi = vec4(texture(u_c0, pe / vec2(size)).rgb * u_gain, 1.0);
}
