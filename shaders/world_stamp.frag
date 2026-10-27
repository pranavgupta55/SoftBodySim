uniform usampler2D u_atlas;  // 1 where a mask pixel is set

flat in ivec2 v_origin;
flat in ivec2 v_atlas;
flat in vec3 v_col;
flat in int v_mat;
uniform sampler2D u_materials;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out uint f_mat;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_clip.x || p.y >= u_clip.y || texelFetch(u_atlas, v_atlas + p - v_origin, 0).r == 0u) {
        discard;
    }
    if (!u_lit || v_mat <= 0) {
        f_color = vec4(v_col, 1.0);
        f_normal = vec4(0.5, 0.5, 0.0, 0.0);
        f_mat = 0u;
        return;
    }
    Mat m = mat_fetch(u_materials, uint(v_mat));  // lighting on: tint and flags, a flat normal
    f_color = vec4(mat_tinted(m, v_col), 1.0);
    f_normal = gbuf_normal(vec2(0.0), 0.0, mat_flags(m));
    f_mat = uint(v_mat);
}
