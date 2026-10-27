// Caustics: each deposited point adds its light (additive blending into an RGBA16F target).
in vec3 v_col;
out vec4 f_color;

void main() {
    f_color = vec4(v_col, 1.0);
}
