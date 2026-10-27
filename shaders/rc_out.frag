// Radiance cascades, output pass (gl_sdf.py GiEffect): cascade 0 (per-texel fluence on the half-res grid E)
// upsampled to the render size as shared["gi"]. Solid texels (bodies) have no light inside them in the cascades:
// - the upsampling is bilinear over the open texels only (a solid one blended in would darken the light next to a
//   body: a dark rim 1-2 px out around every body, emitters included);
// - a body px takes the GI just outside its nearest edge (stepping out along the distance field's gradient): a body
//   is lit on the side facing an emitter. A background px in a solid texel (the 2x2 block it shares with a body's
//   edge) isn't moved: the open texels beside it have its light. Needs sdf_common.glsl.
uniform sampler2D u_c0;     // E, cascade 0 (LINEAR)
uniform sampler2D u_dist;   // E, signed distance in texels (NEAREST)
uniform usampler2D u_mat;   // the G-buffer's material ids (render size): 0 = no body
uniform vec2 u_period;      // texels of E, (0, 0) none
uniform float u_gain;

out vec4 f_gi;

float dist_at(ivec2 q, ivec2 size) {
    if (!sdf_wrap(q, size, u_period)) return 1e4;
    return texelFetch(u_dist, q, 0).r;
}

// bilinear lookup of cascade 0 at E texel coordinates pe, over the open texels; w = their share of the weight
vec3 open_lookup(vec2 pe, ivec2 size, out float w) {
    // wrapped into [0.5, period + 0.5): the lookup always has both neighbours (the grid runs past the period,
    // GiEffect._alloc), so there is no seam at the wrap
    if (u_period.x > 0.0) pe.x -= u_period.x * floor((pe.x - 0.5) / u_period.x);
    if (u_period.y > 0.0) pe.y -= u_period.y * floor((pe.y - 0.5) / u_period.y);
    vec2 f = pe - 0.5;
    ivec2 b = ivec2(floor(f));
    f -= vec2(b);
    vec3 sum = vec3(0.0);
    w = 0.0;
    for (int k = 0; k < 4; k++) {
        ivec2 o = ivec2(k & 1, k >> 1);
        ivec2 t = clamp(b + o, ivec2(0), size - 1);
        float wk = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        if (wk > 0.0 && dist_at(t, size) >= 0.0) {
            sum += wk * texelFetch(u_c0, t, 0).rgb;
            w += wk;
        }
    }
    return w > 1e-4 ? sum / w : vec3(0.0);
}

void main() {
    ivec2 size = textureSize(u_c0, 0);
    vec2 pe = gl_FragCoord.xy * 0.5;  // render px -> E texels
    ivec2 q = ivec2(floor(pe));
    float d = dist_at(q, size);
    float w = 0.0;
    vec3 gi = vec3(0.0);
    bool body = texelFetch(u_mat, ivec2(gl_FragCoord.xy), 0).r != 0u;
    if (d >= 0.0 || !body) gi = open_lookup(pe, size, w);
    if (w <= 1e-4 && d < 0.0) {  // a body px (or a background px with no open texel beside it): step out
        vec2 grad = vec2(dist_at(q + ivec2(1, 0), size) - dist_at(q - ivec2(1, 0), size),
                         dist_at(q + ivec2(0, 1), size) - dist_at(q - ivec2(0, 1), size));
        if (dot(grad, grad) > 0.0) pe += normalize(grad) * (-d + 2.0);
        gi = open_lookup(pe, size, w);
        if (w <= 1e-4) gi = texture(u_c0, pe / vec2(size)).rgb;  // deep in a crowd of bodies: plain bilinear
    }
    f_gi = vec4(gi * u_gain, 1.0);
}
