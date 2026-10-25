// Single pixels (the trails' rasterised samples), in order: a later pixel overwrites an earlier one, as in the numpy
// scatter that trails.py does on the CPU path.
in ivec2 in_pos;
in ivec4 in_col;  // r, g, b, opacity in 256ths

flat out vec4 v_col;

void main() {
    gl_Position = to_clip(vec2(in_pos) + 0.5, 0);
    gl_PointSize = 1.0;
    v_col = vec4(in_col) / 255.0;
}
