// Fluid field, one axis of a separable Gaussian blur (u_dir = (1, 0) then (0, 1)) at half resolution. Run twice:
// the field and its colour (fluid_splat.frag), then (FIELD_ONLY, u_step = threshold) the thresholded liquid mask,
// whose smooth version gives the surface normal and thickness (flat inside the liquid, curved at its rim). Taps wrap
// across a periodic axis and clamp at the view's edge on a walled one (so fluid resting on the edge stays full).
layout(location = 0) out float o_field;
#ifndef FIELD_ONLY
layout(location = 1) out vec4 o_col;
uniform sampler2D u_src_col;
#endif

uniform sampler2D u_src;
uniform ivec2 u_dir;
uniform float u_sigma;   // half-res px
uniform float u_step;    // >= 0: blur step(u_step, field) instead of the field
uniform ivec2 u_hsize;   // half-res size of the view
uniform vec2 u_hperiod;  // half-res period (0 = no wrap)

ivec2 hwrap(ivec2 c) {
    c.x = u_hperiod.x > 0.0 ? int(mod(float(c.x), u_hperiod.x)) : clamp(c.x, 0, u_hsize.x - 1);
    c.y = u_hperiod.y > 0.0 ? int(mod(float(c.y), u_hperiod.y)) : clamp(c.y, 0, u_hsize.y - 1);
    return c;
}

float tap(ivec2 c) {
    float v = texelFetch(u_src, c, 0).r;
    return u_step >= 0.0 ? step(u_step, v) : v;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int rad = min(int(ceil(2.5 * u_sigma)), 16);
    float k = -0.5 / (u_sigma * u_sigma);
    float sum = tap(p), wsum = 1.0;
#ifndef FIELD_ONLY
    vec4 col = texelFetch(u_src_col, p, 0);
#endif
    for (int i = 1; i <= rad; i++) {
        float w = exp(k * float(i * i));
        ivec2 a = hwrap(p + i * u_dir), b = hwrap(p - i * u_dir);
        sum += w * (tap(a) + tap(b));
#ifndef FIELD_ONLY
        col += w * (texelFetch(u_src_col, a, 0) + texelFetch(u_src_col, b, 0));
#endif
        wsum += 2.0 * w;
    }
    o_field = sum / wsum;
#ifndef FIELD_ONLY
    o_col = col / wsum;
#endif
}
