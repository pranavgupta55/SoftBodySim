// The world layer's base, one fullscreen pass: the background, the broad phase grid (World.draw_grid) and the motion
// trails blended over it (trails.py), all in whole 0-255 values so the result matches the CPU path's bytes.
//
// Grid: per column / row of the view, u_xmap / u_ymap hold (cell index, 1 on a grid line); cells are coloured from
// u_cells (0 empty, 1 an edge's padded box, 2 a point), then lines go on top, as draw_grid does with fills and lines.
// Trails: u_trail holds each trail pixel's colour and opacity a (256ths, in alpha); the CPU path blends
// pixel = (pixel * (256 - a) + colour * a) >> 8.
// G-buffer (lighting on): all of this is albedo; normal flat with no flags, material 0.
uniform vec3 u_background;  // 0-255
uniform bool u_grid;
uniform isampler2D u_xmap;
uniform isampler2D u_ymap;
uniform usampler2D u_cells;
uniform vec3 u_line_col;   // 0-255
uniform vec3 u_edge_col;
uniform vec3 u_point_col;
uniform bool u_trails;
uniform sampler2D u_trail;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out uint f_mat;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 col = u_background;
    if (p.x < u_clip.x && p.y < u_clip.y) {
        if (u_grid) {
            ivec2 cx = texelFetch(u_xmap, ivec2(p.x, 0), 0).xy;
            ivec2 cy = texelFetch(u_ymap, ivec2(p.y, 0), 0).xy;
            uint kind = texelFetch(u_cells, ivec2(cx.x, cy.x), 0).r;
            if (kind == 1u) {
                col = u_edge_col;
            } else if (kind == 2u) {
                col = u_point_col;
            }
            if (cx.y == 1 || cy.y == 1) {
                col = u_line_col;
            }
        }
        if (u_trails) {
            vec4 t = texelFetch(u_trail, p, 0);
            float a = floor(t.a * 255.0 + 0.5);
            if (a > 0.0) {
                col = floor((col * (256.0 - a) + floor(t.rgb * 255.0 + 0.5) * a) / 256.0);
            }
        }
    }
    f_color = vec4(col / 255.0, 1.0);
    f_normal = vec4(0.5, 0.5, 0.0, 0.0);  // background: flat, no body, no flags
    f_mat = 0u;
}
