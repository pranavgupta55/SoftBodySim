// gl_lights.py's fullscreen passes: a 4-vertex triangle strip over the whole target from gl_VertexID.
void main() {
    gl_Position = vec4((gl_VertexID & 1) == 1 ? 1.0 : -1.0, (gl_VertexID & 2) == 2 ? 1.0 : -1.0, 0.0, 1.0);
}
