// One pygame.draw.circle mask stamped (world_stamp.vert). v_mat: the material id, plus STAMP_BORDER for a black
// border part (a segment body's round caps, a hook's black disc and hole): FLAG_BORDER, not lit, like border lines.
// u_seethrough / u_flags_only / u_points: gl_world's lit passes (world_fill.frag, world_line.frag).
uniform usampler2D u_atlas;  // 1 where a mask pixel is set
uniform int u_seethrough;
uniform bool u_flags_only;
uniform bool u_points;
const int STAMP_BORDER = 1 << 24;  // gl_world.STAMP_BORDER

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
    int id = v_mat & (STAMP_BORDER - 1);
    if (!u_lit || id <= 0) {
        f_color = vec4(v_col, 1.0);
        f_normal = vec4(0.5, 0.5, 0.0, 0.0);
        f_mat = 0u;
        return;
    }
    Mat m = mat_fetch(u_materials, uint(id));  // lighting on: tint and flags, a flat normal
    bool see = m.kind == KIND_GLASS || m.kind == KIND_FLUID;
    if ((u_seethrough == 1 && see) || (u_seethrough == 2 && !see)) {
        discard;
    }
    if (u_flags_only) {
        f_color = vec4(0.0);
        f_normal = vec4(0.0, 0.0, 0.0, float(mat_flags(m) | FLAG_LIT) / 255.0);
        f_mat = 0u;
        return;
    }
    int flags = mat_flags(m);
    if (u_points) {
        flags &= FLAG_LIT;
    } else if ((v_mat & STAMP_BORDER) != 0) {
        flags = (flags & ~FLAG_LIT) | FLAG_BORDER;
    }
    f_color = vec4(mat_tinted(m, v_col), 1.0);
    f_normal = gbuf_normal(vec2(0.0), 0.0, flags);
    f_mat = uint(id);
}
