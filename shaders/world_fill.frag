// pygame.draw.polygon's scanline fill (draw_fillpoly), pixel for pixel, for whole-pixel vertices. pygame, per row y:
// every non-horizontal edge (upper end y1, lower end y2) with y1 <= y < y2, or ending on the bottom row (y == maxy
// == y2), crosses it at x1 + floor((y - y1) * (x2 - x1) / (y2 - y1)); the crossings are sorted and each pair
// (0, 1), (2, 3), ... is filled inclusive. Horizontal edges strictly between the top and bottom rows are filled too,
// and a polygon one row high is the span of its x range. Instead of sorting, a pixel counts the crossings left of it
// (L) and on it (E): it is in a pair's span iff L is odd and crossing L has a partner, or E > 0 and the crossing at
// L starts a pair.
uniform isampler2D u_verts;  // whole-pixel vertices, vertex i at texel (i % VERT_ROW, i / VERT_ROW)

flat in ivec4 v_info;
flat in ivec4 v_box;
flat in vec3 v_col;

out vec4 f_color;

const int VERT_ROW = 4096;

ivec2 vert(int i) {
    return texelFetch(u_verts, ivec2(i % VERT_ROW, i / VERT_ROW), 0).xy;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_clip.x || p.y >= u_clip.y) {
        discard;
    }
    int miny = v_box.y, maxy = v_box.w;
    if (miny != maxy) {  // (one row high: the whole box is the span)
        int start = v_info.x, n = v_info.y;
        int left = 0, on = 0, crossings = 0;
        bool flat_edge = false;
        ivec2 a = vert(start + n - 1);
        for (int i = 0; i < n; i++) {
            ivec2 b = vert(start + i);
            ivec2 top = a.y < b.y ? a : b, bottom = a.y < b.y ? b : a;
            if (a.y == b.y) {
                flat_edge = flat_edge || (p.y == a.y && miny < p.y && p.y < maxy && p.x >= min(a.x, b.x)
                                          && p.x <= max(a.x, b.x));
            } else if ((p.y >= top.y && p.y < bottom.y) || (p.y == maxy && bottom.y == maxy)) {
                int x = top.x + floordiv((p.y - top.y) * (bottom.x - top.x), bottom.y - top.y);
                crossings++;
                left += int(x < p.x);
                on += int(x == p.x);
            }
            a = b;
        }
        int paired = crossings - crossings % 2;  // an odd last crossing has no partner
        bool inside = (left % 2 == 1) ? left < paired : (on > 0 && left + 1 < paired);
        if (!inside && !flat_edge) {
            discard;
        }
    }
    f_color = vec4(v_col, 1.0);
}
