// Caustics: one axis of a separable Gaussian blur (sigma CAUSTIC_BLUR_PX, wrapping on periodic axes) of the
// deposited points; run along x then y -> shared["caustics"]. The final (y) pass also soft-caps the result: a focus
// where thousands of rays meet would otherwise add tens of times the light's own irradiance (a clipped white blob).
// Below CAUSTIC_KNEE nothing changes (light merely passing through glass stays exact); above it the brightest
// channel rolls off smoothly toward CAUSTIC_CAP, hue kept.
#ifndef CAUSTIC_CAP
#define CAUSTIC_CAP 4.0
#endif
#ifndef CAUSTIC_KNEE
#define CAUSTIC_KNEE 2.0
#endif
out vec4 f_color;

vec3 soft_cap(vec3 c) {
    float m = max(max(c.r, c.g), c.b);
    if (m <= CAUSTIC_KNEE) return c;
    float x = m - CAUSTIC_KNEE, room = CAUSTIC_CAP - CAUSTIC_KNEE;
    return c * ((CAUSTIC_KNEE + x / (1.0 + x / room)) / m);
}

uniform sampler2D u_src;
uniform ivec2 u_dir;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int rad = int(ceil(2.5 * CAUSTIC_BLUR_PX));
    float k = -0.5 / (CAUSTIC_BLUR_PX * CAUSTIC_BLUR_PX);
    vec3 sum = texelFetch(u_src, p, 0).rgb;
    float wsum = 1.0;
    for (int i = 1; i <= rad; i++) {
        float w = exp(k * float(i * i));
        sum += w * (texelFetch(u_src, pixi(p + i * u_dir), 0).rgb + texelFetch(u_src, pixi(p - i * u_dir), 0).rgb);
        wsum += 2.0 * w;
    }
    vec3 c = sum / wsum;
    f_color = vec4(u_dir.y != 0 ? soft_cap(c) : c, 1.0);
}
