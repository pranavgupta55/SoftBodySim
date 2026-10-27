// pygame.draw.polygon's scanline fill (draw_fillpoly), pixel for pixel, for whole-pixel vertices. pygame, per row y:
// every non-horizontal edge (upper end y1, lower end y2) with y1 <= y < y2, or ending on the bottom row (y == maxy
// == y2), crosses it at x1 + floor((y - y1) * (x2 - x1) / (y2 - y1)); the crossings are sorted and each pair
// (0, 1), (2, 3), ... is filled inclusive. Horizontal edges strictly between the top and bottom rows are filled too,
// and a polygon one row high is the span of its x range. Instead of sorting, a pixel counts the crossings left of it
// (L) and on it (E): it is in a pair's span iff L is odd and crossing L has a partner, or E > 0 and the crossing at
// L starts a pair.
//
// Lighting on (u_lit), the same edge loop also finds, for the G-buffer:
//   normal   the distance d to the nearest edge and the inward direction e from it (toward the pixel; the edge's
//            inward perpendicular on the edge itself or for a pixel just outside), d rounded down to whole px (+0.5)
//            for a stepped pixel-art bevel; height profile h(d) from the material (mat_slope), N = norm(-h'(d) e, 1).
//            e is a soft-min blend of every edge's direction, w_i = exp(-(d_i - d_min) / TAU), TAU = max(1.5 px,
//            0.25 bevel_px) (the sum rescaled whenever the running minimum drops), so a polygon's bevel rounds off
//            instead of shading as flat pie wedges with creases on the medial axis; d stays the nearest edge's
//            distance (the slope)
//   pattern  the pixel's rest-frame position: mean value coordinates (Hormann & Floater 2006) of the per-vertex
//            rest positions in u_uvs (world units, gl_world's _rest_coords), so patterns stretch and turn with the
//            body. Per edge (a, b) with s = vertex - pixel, r = |s|: t = tan(angle(a, b) / 2) = (r_a r_b - s_a.s_b)
//            / (s_a x s_b) (signed, so concave loops work); vertex a gets t / r_a and b gets t / r_b. A pixel on a
//            vertex or an edge takes that vertex's / the edge's interpolated position directly.
uniform isampler2D u_verts;  // whole-pixel vertices, vertex i at texel (i % VERT_ROW, i / VERT_ROW)
uniform sampler2D u_uvs;     // rest-frame position (world units) of vertex i, same layout (lighting on)
uniform sampler2D u_materials;
uniform float u_zoom;
// gl_world's lit passes (world_line / world_stamp have the same two):
//   u_seethrough   0 draw everything; 1 (pass A) discard GLASS / FLUID bodies; 2 (pass B) draw only those
//   u_flags_only   the points view's invisible occluder fill: only normal.a is written (gl_world masks the other
//                  outputs): the filled body's occluder / casts-shadow flags plus FLAG_LIT, so shadows match the
//                  filled view's
uniform int u_seethrough;
uniform bool u_flags_only;

flat in ivec4 v_info;
flat in ivec4 v_box;
flat in vec3 v_col;
flat in ivec2 v_mat;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out uint f_mat;

const int VERT_ROW = 4096;

ivec2 vert(int i) {
    return texelFetch(u_verts, ivec2(i % VERT_ROW, i / VERT_ROW), 0).xy;
}

vec2 rest(int i) {
    return texelFetch(u_uvs, ivec2(i % VERT_ROW, i / VERT_ROW), 0).xy;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_clip.x || p.y >= u_clip.y) {
        discard;
    }
    int miny = v_box.y, maxy = v_box.w;
    bool test = miny != maxy;  // (one row high: the whole box is the span)
    // lighting on: nearest edge, mean value coordinates
    vec2 x = vec2(p);
    float best = 1e30;
    vec2 best_dir = vec2(0.0), best_uv = vec2(0.0), uv_sum = vec2(0.0), exact_uv = vec2(0.0);
    float w_sum = 0.0;
    bool exact = false;
    Mat m;
    bool full = u_lit && !u_flags_only;  // normals and pattern coordinates wanted
    float tau = 0.0, soft_min = 1e30;
    vec2 soft_dir = vec2(0.0);
    if (u_lit) {
        m = mat_fetch(u_materials, uint(v_mat.x));
        bool see = m.kind == KIND_GLASS || m.kind == KIND_FLUID;
        if ((u_seethrough == 1 && see) || (u_seethrough == 2 && !see)) {
            discard;
        }
        tau = m.bevel_px > 0.0 ? max(1.5, 0.25 * m.bevel_px) : 0.0;  // 0: flat, no normal blend needed
    }
    if (test || full) {
        int start = v_info.x, n = v_info.y;
        int left = 0, on = 0, crossings = 0;
        bool flat_edge = false;
        int ia = start + n - 1;
        ivec2 a = vert(ia);
        vec2 sa = vec2(a) - x, uva = vec2(0.0);
        float ra = length(sa);
        if (full) {
            uva = rest(ia);
        }
        for (int i = 0; i < n; i++) {
            ivec2 b = vert(start + i);
            ivec2 top = a.y < b.y ? a : b, bottom = a.y < b.y ? b : a;
            if (a.y == b.y) {
                flat_edge = flat_edge || (p.y == a.y && miny < p.y && p.y < maxy && p.x >= min(a.x, b.x)
                                          && p.x <= max(a.x, b.x));
            } else if ((p.y >= top.y && p.y < bottom.y) || (p.y == maxy && bottom.y == maxy)) {
                int cx = top.x + floordiv((p.y - top.y) * (bottom.x - top.x), bottom.y - top.y);
                crossings++;
                left += int(cx < p.x);
                on += int(cx == p.x);
            }
            if (full) {
                vec2 uvb = rest(start + i);
                vec2 sb = vec2(b) - x;
                float rb = length(sb);
                // nearest edge
                vec2 e = vec2(b - a);
                float l2 = dot(e, e);
                float t = l2 > 0.0 ? clamp(dot(-sa, e) / l2, 0.0, 1.0) : 0.0;
                vec2 dv = -sa - t * e;  // from the closest point on the edge to the pixel
                float d2 = dot(dv, dv);
                if (l2 > 0.0) {
                    vec2 inward = -float(v_mat.y) * vec2(e.y, -e.x) / sqrt(l2);
                    vec2 dir = (d2 > 1e-6 && dot(dv, inward) > 0.0) ? dv * inversesqrt(d2) : inward;
                    if (d2 < best) {
                        best = d2;
                        best_dir = dir;
                        best_uv = mix(uva, uvb, t);
                    }
                    if (tau > 0.0) {  // soft-min of the edge directions
                        float di = sqrt(d2);
                        if (di < soft_min) {
                            soft_dir *= exp((di - soft_min) / tau);
                            soft_min = di;
                        }
                        float g = (di - soft_min) / tau;
                        if (g < 12.0) {
                            soft_dir += exp(-g) * dir;
                        }
                    }
                }
                // mean value coordinates
                if (rb < 1e-4) {
                    exact = true;
                    exact_uv = uvb;
                } else if (ra >= 1e-4) {
                    float cr = sa.x * sb.y - sa.y * sb.x, dt = dot(sa, sb);
                    if (abs(cr) < 1e-5) {
                        if (dt < 0.0) {  // on the edge, between a and b
                            exact = true;
                            exact_uv = mix(uva, uvb, ra / (ra + rb));
                        }
                    } else {
                        float tn = (ra * rb - dt) / cr;
                        w_sum += tn / ra + tn / rb;
                        uv_sum += tn * (uva / ra + uvb / rb);
                    }
                }
                sa = sb;
                ra = rb;
                uva = uvb;
            }
            a = b;
        }
        if (test) {
            int paired = crossings - crossings % 2;  // an odd last crossing has no partner
            bool inside = (left % 2 == 1) ? left < paired : (on > 0 && left + 1 < paired);
            if (!inside && !flat_edge) {
                discard;
            }
        }
    }
    if (!u_lit) {
        f_color = vec4(v_col, 1.0);
        f_normal = vec4(0.0);
        f_mat = 0u;
        return;
    }
    if (u_flags_only) {
        f_color = vec4(0.0);
        f_normal = vec4(0.0, 0.0, 0.0, float(mat_flags(m) | FLAG_LIT) / 255.0);
        f_mat = 0u;
        return;
    }
    if (tau > 0.0 && dot(soft_dir, soft_dir) > 1e-8) {
        best_dir = normalize(soft_dir);
    }
    vec2 uv = exact ? exact_uv : (abs(w_sum) > 1e-6 ? uv_sum / w_sum : best_uv);
    float d = best < 1e29 ? sqrt(best) : 0.0;
    float slope = mat_slope(m, floor(d) + 0.5);
    vec3 nrm = normalize(vec3(-slope * best_dir, 1.0));
    f_color = vec4(mat_patterned(m, mat_tinted(m, v_col), uv * u_zoom), 1.0);
    f_normal = gbuf_normal(nrm.xy, d, mat_flags(m));
    f_mat = uint(v_mat.x);
}
