// Caustics (gl_glass.CausticsEffect): one input point per ray (gl_PrimitiveIDIn = ray index). Each light shoots
// u_rays rays in the plane, evenly spaced (point: all around; spot: across its cone; directional: parallel, across
// the view). A ray is marched once (glass_common.march: Snell, TIR, mirrors) and every straight piece of its path
// that runs in open air after a refraction (or after any reflection, with CAUSTIC_MIRRORS) is drawn as a line,
// additively, into the caustics target. Lines rather than points sampled along the ray: the deposit is continuous
// along the path, so ray density alone sets the brightness (no dotted / grainy rays), and it's deterministic, so
// it is stable from frame to frame.
//
// Weight: a pixel crossed by a line gets one fragment per pixel step along the line's major axis, i.e.
// max(|u.x|, |u.y|) fragments per px of path, hence the 1 / max(|u.x|, |u.y|). Rays t * dTheta apart (spacing * t;
// directional: spacing) then give, before any lens, the light's own unshadowed irradiance colour x falloff(t) per
// pixel, as the light pass would; converging rays pile up into caustics.
layout(points) in;
layout(line_strip, max_vertices = CAUSTIC_MAX_VERTS) out;

uniform int u_kind;          // 0 point, 1 spot, 2 directional
uniform vec2 u_light_pos;    // px
uniform vec2 u_light_dir;
uniform float u_cone;        // spot half angle, radians
uniform vec3 u_light_col;    // colour x intensity x CAUSTIC_GAIN
uniform float u_radius_px;
uniform float u_falloff;
uniform int u_rays;
uniform float u_max_t;
uniform vec2 u_size;

out vec3 v_col;

float g_spacing;  // px between neighbouring rays (per px of path for point / spot)
vec2 g_start;     // where the current straight piece began
float g_t0;       // path length there
int g_verts = 0;  // vertices emitted so far

vec3 weight(float t, vec3 thr) {
    float fall = u_kind == 2 ? 1.0 : pow(max(1.0 - t / u_radius_px, 0.0), u_falloff);
    return u_light_col * thr * fall * (u_kind == 2 ? g_spacing : g_spacing * max(t, 1.0));
}

void vert(vec2 q, vec3 col) {
    v_col = col;
    gl_Position = vec4(q / u_size * 2.0 - 1.0, 0.0, 1.0);
    EmitVertex();
}

// One sub-segment a -> b (unwrapped px) as its own 2-vertex strip, shifted into the periodic tile, plus a copy
// shifted back by a period on each axis it runs out of the tile on.
void segment(vec2 a, vec2 b, vec3 ca, vec3 cb) {
    vec2 sh = vec2(0.0);
    vec2 cp = vec2(0.0);  // the copy's extra shift per axis (0 = no copy)
    for (int k = 0; k < 2; k++) {
        if (u_period[k] > 0.0) {
            sh[k] = -u_period[k] * floor(a[k] / u_period[k]);
            float e = b[k] + sh[k];
            cp[k] = e < 0.0 ? u_period[k] : (e >= u_period[k] ? -u_period[k] : 0.0);
        }
    }
    for (int i = 0; i < 4; i++) {
        vec2 s = vec2((i & 1) == 1 ? cp.x : 0.0, (i & 2) == 2 ? cp.y : 0.0);
        if (((i & 1) == 1 && cp.x == 0.0) || ((i & 2) == 2 && cp.y == 0.0)) continue;
        if (g_verts + 2 > CAUSTIC_MAX_VERTS) return;
        vert(a + sh + s, ca);
        vert(b + sh + s, cb);
        EndPrimitive();
        g_verts += 2;
    }
}

void caustic_piece(vec2 at, vec3 thr, int events, int refr, bool inside, float t) {
    vec2 a = g_start;
    float ta = g_t0;
    g_start = at;
    g_t0 = t;
    if (inside || (CAUSTIC_MIRRORS ? events : refr) == 0) return;
    vec2 d = at - a;
    float len = length(d);
    if (len < 0.25) return;
    vec2 u = d / len;
    float axis = 1.0 / max(abs(u.x), abs(u.y));
    int n = int(ceil(len / CAUSTIC_SEG_PX));
    if (u_period == vec2(0.0)) {  // one strip of n + 1 vertices
        n = min(n, CAUSTIC_MAX_VERTS - g_verts - 1);
        if (n < 1) return;
        for (int i = 0; i <= n; i++) {
            float f = float(i) / float(n);
            vert(a + d * f, weight(mix(ta, t, f), thr) * axis);
        }
        EndPrimitive();
        g_verts += n + 1;
        return;
    }
    vec2 p0 = a;
    vec3 c0 = weight(ta, thr) * axis;
    for (int i = 1; i <= n; i++) {
        float f = float(i) / float(n);
        vec2 p1 = a + d * f;
        vec3 c1 = weight(mix(ta, t, f), thr) * axis;
        segment(p0, p1, c0, c1);
        p0 = p1;
        c0 = c1;
    }
}

void main() {
    int ray = gl_PrimitiveIDIn;
    float u = (float(ray) + 0.5) / float(u_rays);
    float diag = length(vec2(u_clip));
    vec2 o, dir;
    if (u_kind == 2) {
        dir = normalize(u_light_dir);
        vec2 perp = vec2(-dir.y, dir.x);
        o = 0.5 * vec2(u_clip) - dir * 0.5 * diag + perp * (u - 0.5) * diag;
        g_spacing = diag / float(u_rays);
    } else {
        float a0 = atan(u_light_dir.y, u_light_dir.x);
        float a = u_kind == 1 ? a0 + (2.0 * u - 1.0) * u_cone : 6.2831853 * u;
        dir = vec2(cos(a), sin(a));
        o = u_light_pos;
        g_spacing = (u_kind == 1 ? 2.0 * u_cone : 6.2831853) / float(u_rays);
    }
    g_start = o;
    g_t0 = 0.0;
    march(o, dir, u_max_t, 0u);
}
