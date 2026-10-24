#version 330 core

in vec2 in_vert;
in vec2 in_text;

out vec2 v_text;

void main() {
    v_text = in_text;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
