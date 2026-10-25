uniform usampler2D u_atlas;  // 1 where a mask pixel is set

flat in ivec2 v_origin;
flat in ivec2 v_atlas;
flat in vec3 v_col;

out vec4 f_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_clip.x || p.y >= u_clip.y || texelFetch(u_atlas, v_atlas + p - v_origin, 0).r == 0u) {
        discard;
    }
    f_color = vec4(v_col, 1.0);
}
