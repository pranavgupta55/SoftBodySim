// Caustics: one 7x7 binomial blur pass over the deposited points (wrapping on periodic axes) -> shared["caustics"].
out vec4 f_color;

uniform sampler2D u_src;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    const float w[7] = float[7](1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0);
    vec3 sum = vec3(0.0);
    for (int y = -3; y <= 3; y++)
        for (int x = -3; x <= 3; x++)
            sum += w[x + 3] * w[y + 3] * texelFetch(u_src, pixi(p + ivec2(x, y)), 0).rgb;
    f_color = vec4(sum / 4096.0, 1.0);
}
