#version 330 core

out vec4 f_color;
in vec2 v_text;

uniform sampler2D Texture;

void main() {
    // We do not need to flip Y here because we defined the UVs 
    // in the Python quad_buffer to match Pygame's coordinate system.
    // (Top-Left 0,0)
    f_color = texture(Texture, v_text);
}
