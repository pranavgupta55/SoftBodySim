// One instance per pygame.draw.line: a quad over its pixel box (see world_line.frag for which pixels it keeps).
in ivec4 in_ends;   // x1, y1, x2, y2: whole pixels (pygame truncates float endpoints)
in ivec4 in_style;  // width px, packed 0xBBGGRR colour, depth layer,
                    // transparency 0..255 (0 = opaque)
in ivec2 in_mat;    // material id (-1: an overlay, drawn after the lit effects), kind: 0 flat normal, +1 / -1 a
                    // body's rim (no-outline view) with that winding (normal tilted outward like a bevel's edge),
                    // 2 a body's black border (flat normal, FLAG_NO_EMIT: an emissive body's glow stops at its fill)

flat out ivec4 v_ends;
flat out int v_width;
flat out vec3 v_col;
flat out float v_alpha;
flat out ivec2 v_mat;

void main() {
    int pad = in_style.x / 2 + 1;
    ivec2 lo = min(in_ends.xy, in_ends.zw) - pad;
    ivec2 hi = max(in_ends.xy, in_ends.zw) + pad + 1;
    gl_Position = to_clip(box_corner(lo, hi), in_style.z);
    v_ends = in_ends;
    v_width = in_style.x;
    v_col = unpack_rgb(in_style.y);
    v_alpha = 1.0 - float(in_style.w) / 255.0;
    v_mat = in_mat;
}
