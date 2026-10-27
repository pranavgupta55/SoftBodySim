// Lights pass (gl_lights.py): one instance per light, a 4-vertex triangle strip covering the pixels it can reach.
// Point lights: the box of their radius; spot lights: the box of their cone sector; directional lights: the whole
// surface. On a periodic axis a box that crosses the seam widens to the whole axis, so the fragment shader's
// minimum-image distance lights the far side too.
layout(std140) uniform Lights {
    vec4 u_lights[1024];  // 4 vec4 per light (gl_lights.pack_lights)
};

uniform vec2 u_res;      // surface size, px
uniform vec2 u_period;   // periodic span per axis, px (0 = not periodic)

flat out int v_light;

void main() {
    int i = gl_InstanceID;
    v_light = i;
    vec4 a = u_lights[4 * i];      // pos.xy, height, kind
    vec4 b = u_lights[4 * i + 1];  // colour * intensity, radius
    vec4 c = u_lights[4 * i + 2];  // dir.xy, cos inner, cos outer
    int kind = int(a.w + 0.5);
    vec2 lo = vec2(0.0), hi = u_res;
    if (kind != 2) {
        float r = b.w;
        lo = a.xy - r;
        hi = a.xy + r;
        if (kind == 1 && c.w > 0.0) {  // a cone narrower than a half plane: the box of its sector
            vec2 d = c.xy;
            float so = sqrt(max(1.0 - c.w * c.w, 0.0));
            vec2 e1 = a.xy + r * vec2(d.x * c.w - d.y * so, d.x * so + d.y * c.w);
            vec2 e2 = a.xy + r * vec2(d.x * c.w + d.y * so, -d.x * so + d.y * c.w);
            lo = min(a.xy, min(e1, e2));
            hi = max(a.xy, max(e1, e2));
            // the arc's axis extremes it passes through
            if (d.x >= c.w) hi.x = a.x + r;
            if (-d.x >= c.w) lo.x = a.x - r;
            if (d.y >= c.w) hi.y = a.y + r;
            if (-d.y >= c.w) lo.y = a.y - r;
        }
        for (int k = 0; k < 2; k++) {
            if (u_period[k] > 0.0 && (lo[k] < 0.0 || hi[k] > u_period[k])) {
                lo[k] = 0.0;
                hi[k] = u_res[k];
            }
        }
        lo = clamp(floor(lo), vec2(0.0), u_res);
        hi = clamp(ceil(hi), vec2(0.0), u_res);
    }
    vec2 px = vec2((gl_VertexID & 1) == 1 ? hi.x : lo.x, (gl_VertexID & 2) == 2 ? hi.y : lo.y);
    gl_Position = vec4(px / u_res * 2.0 - 1.0, 0.0, 1.0);
}
