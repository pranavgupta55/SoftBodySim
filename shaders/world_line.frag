// pygame.draw.line, pixel for pixel. pygame steps a Bresenham line from (x1, y1) to (x2, y2) with
// err = (dx > dy ? dx : -dy) / 2; the pixel it plots at step k along the major axis is offset along the minor axis by
// ceil((k * minor - major / 2) / major) (derived from that loop and checked against pygame for every width). Width
// w > 1 repeats the line shifted along the minor axis by -(w / 2) + (1 - w % 2) .. w / 2 (pygame's draw_line_width).
// Lighting on: the body's material (tint, flags); a flat normal except on rim lines; border lines get FLAG_BORDER and
// drop FLAG_LIT (a black outline stays black: no light, specular or glow on it; still an occluder / shadow caster).
// u_seethrough / u_flags_only: gl_world's lit passes (world_fill.frag). u_points (the points view): sticks are lit
// but neither occluders nor shadow casters (the invisible fill of the body silhouettes, u_flags_only, casts instead).
flat in ivec4 v_ends;
flat in int v_width;
flat in vec3 v_col;
flat in float v_alpha;
flat in ivec2 v_mat;
uniform sampler2D u_materials;
uniform int u_seethrough;
uniform bool u_flags_only;
uniform bool u_points;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out uint f_mat;
layout(location = 3) out vec4 f_tr;  // u_flags_only: gbuf["tr"] (world_fill.frag)

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_clip.x || p.y >= u_clip.y) {
        discard;
    }
    int x1 = v_ends.x, y1 = v_ends.y, x2 = v_ends.z, y2 = v_ends.w;
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    int sx = x2 > x1 ? 1 : -1, sy = y2 > y1 ? 1 : -1;
    int off;
    if (dx > dy) {  // x major: one pixel per column, thickness grows in y
        int k = (p.x - x1) * sx;
        if (k < 0 || k > dx) {
            discard;
        }
        off = p.y - (y1 + sy * ceildiv(k * dy - dx / 2, dx));
    } else {  // y major (or a single point): one pixel per row, thickness grows in x
        int k = (p.y - y1) * sy;
        if (k < 0 || k > dy) {
            discard;
        }
        off = p.x - (dy == 0 ? x1 : x1 + sx * ceildiv(k * dx - dy / 2, dy));
    }
    int half_w = v_width / 2;
    if (off < -half_w + (1 - v_width % 2) || off > half_w) {
        discard;
    }
    if (!u_lit || v_mat.x <= 0) {  // (overlays: drawn after the effects, plain)
        f_color = vec4(v_col, v_alpha);
        f_normal = vec4(0.5, 0.5, 0.0, 0.0);
        f_mat = 0u;
        return;
    }
    Mat m = mat_fetch(u_materials, uint(v_mat.x));
    bool see = m.kind == KIND_GLASS || m.kind == KIND_FLUID;
    if ((u_seethrough == 1 && see) || (u_seethrough == 2 && !see)) {
        discard;
    }
    if (u_flags_only) {
        f_color = vec4(0.0);
        f_normal = vec4(0.0, 0.0, 0.0, float(mat_flags(m) | FLAG_LIT) / 255.0);
        f_mat = 0u;
        f_tr = mat_tr_px(m, mat_tinted(m, v_col));  // (an outline comes with its body's colour: gl_world)
        return;
    }
    vec2 n = vec2(0.0);
    if (abs(v_mat.y) == 1 && (dx | dy) != 0) {  // a rim line: the outward perpendicular (draw_border's), at the edge's slope
        vec2 e = normalize(vec2(x2 - x1, y2 - y1));
        n = float(v_mat.y) * vec2(e.y, -e.x) * mat_slope(m, 0.5);
    }
    f_color = vec4(mat_tinted(m, v_col), v_alpha);
    int flags = mat_flags(m);
    if (u_points) {
        flags &= FLAG_LIT;
    } else if (v_mat.y == 2) {
        flags = (flags & ~FLAG_LIT) | FLAG_BORDER;
    }
    f_normal = gbuf_normal(normalize(vec3(n, 1.0)).xy, 0.0, flags);
    f_mat = uint(v_mat.x);
}
