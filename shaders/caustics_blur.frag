// Caustics: one axis of a separable Gaussian blur (sigma CAUSTIC_BLUR_PX, wrapping on periodic axes) of the
// deposited points; run along x then y -> shared["caustics"].
out vec4 f_color;

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
    f_color = vec4(sum / wsum, 1.0);
}
