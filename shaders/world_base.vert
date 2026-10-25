// A fullscreen triangle strip from gl_VertexID (no vertex buffer).
void main() {
    gl_Position = vec4((gl_VertexID & 1) == 1 ? 1.0 : -1.0, (gl_VertexID & 2) == 2 ? 1.0 : -1.0, 0.0, 1.0);
}
