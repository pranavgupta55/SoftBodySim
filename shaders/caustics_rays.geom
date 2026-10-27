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
// pixel, as the light pass would; converging rays pile up into caustics. Directional rays start where they enter
// the view (clip_directional), so their whole path length is spent on screen.
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
float g_t_skip = 0.0;  // path length before which nothing is deposited (a periodic sun's warm-up lap)
float g_fade_t;        // the deposit fades out over the last quarter of this path length

vec3 weight(float t, vec3 thr) {
    float fall = u_kind == 2 ? 1.0 : pow(max(1.0 - t / u_radius_px, 0.0), u_falloff);
    // the path cap: fade the deposit out over its last quarter, not a hard cut across the beam
    fall *= 1.0 - smoothstep(0.75 * g_fade_t, g_fade_t, t);
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
    if (inside || (CAUSTIC_MIRRORS ? events : refr) == 0 || t <= g_t_skip) return;
    if (ta < g_t_skip) {  // the piece starts in the warm-up: keep the part after it
        a = mix(a, at, (g_t_skip - ta) / (t - ta));
        ta = g_t_skip;
    }
    ta -= g_t_skip;
    t -= g_t_skip;
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

bool finite2(vec2 v) {
    return !any(isnan(v)) && !any(isinf(v));
}

// A directional ray: the line o0 + dir * s (s in [0, diag], o0 0.5 * diag before the view centre, across the strip
// the light covers), clipped to the view box. Returns false if this ray doesn't light the view; else its entry
// point and how far it runs before it leaves the view. Entry is through the box's non-periodic edges facing the
// light, and a ray is kept only if its entry point is inside the period on the periodic axes (one copy of each
// wrapped ray, so nothing is lit twice). With no non-periodic edge to enter by (a fully periodic map, or a ray
// running along the non-periodic walls), rays enter at the view edge of the periodic axis the light runs most
// along, and `run` is one lap: the path length until the ray is back at that edge (each pixel is crossed once).
// `lapped` says which case it is.
bool clip_directional(vec2 o0, vec2 dir, float diag, out vec2 entry, out float run, out bool lapped) {
    vec2 box = vec2(u_clip);
    for (int k = 0; k < 2; k++) if (u_period[k] > 0.0) box[k] = u_period[k];
    float t_in = 0.0, t_out = diag;
    bool bounded = false;  // entered through a non-periodic edge
    int major = abs(dir.x) >= abs(dir.y) ? 0 : 1;
    for (int k = 0; k < 2; k++) {
        bool periodic = u_period[k] > 0.0;
        if (abs(dir[k]) < 1e-6) {
            if (!periodic && (o0[k] < 0.0 || o0[k] > box[k])) return false;
            continue;
        }
        float a = (0.0 - o0[k]) / dir[k], b = (box[k] - o0[k]) / dir[k];
        if (!periodic) {
            t_in = max(t_in, min(a, b));
            t_out = min(t_out, max(a, b));
            bounded = true;
        }
    }
    if (!bounded) {
        float a = (0.0 - o0[major]) / dir[major], b = (box[major] - o0[major]) / dir[major];
        t_in = max(t_in, min(a, b));
    }
    if (bounded && t_out <= t_in + 0.5) return false;
    entry = o0 + dir * t_in;
    for (int k = 0; k < 2; k++) {
        if (u_period[k] > 0.0 && (bounded || k != major) && (entry[k] < 0.0 || entry[k] >= box[k])) return false;
    }
    lapped = !bounded;
    run = bounded ? t_out - t_in : box[major] / abs(dir[major]);
    return true;
}

void main() {
    // garbage lights (Python sanitises them too): draw nothing rather than NaN lines
    if (!finite2(u_light_pos) || !finite2(u_light_dir) || isnan(u_falloff) || isinf(u_falloff)) return;
    vec2 ldir = dot(u_light_dir, u_light_dir) > 1e-12 ? normalize(u_light_dir) : vec2(0.0, 1.0);
    int ray = gl_PrimitiveIDIn;
    float u = (float(ray) + 0.5) / float(u_rays);
    float diag = length(vec2(u_clip));
    float max_t = u_max_t;
    g_fade_t = u_max_t;
    vec2 o, dir;
    if (u_kind == 2) {
        dir = ldir;
        vec2 perp = vec2(-dir.y, dir.x);
        g_spacing = diag / float(u_rays);
        vec2 entry;
        float run;
        bool lapped;
        if (!clip_directional(0.5 * vec2(u_clip) - dir * 0.5 * diag + perp * (u - 0.5) * diag, dir, diag, entry, run,
                              lapped))
            return;
        if (lapped) {
            // A periodic sun: the light reaching the entry edge has already crossed the copies of everything
            // upstream. March one warm-up lap from a period back (the same point, wrapped), depositing nothing, so
            // the ray carries the refractions it picked up on the way in; then deposit over the next lap only,
            // all of it (the entry edge is arbitrary, so no u_max_t fade from it).
            g_t_skip = run;
            o = entry - dir * run;
            max_t = 2.0 * run;
            g_fade_t = 1e9;
            g_extra_steps = int(ceil(run / STEP_MAX_PX)) + 8;  // the warm-up lap's air steps
        } else {
            o = entry + dir * 0.5;  // half a px inside, so the march doesn't start off the view
            max_t = min(max_t, run - 0.5);
        }
    } else {
        float a0 = atan(ldir.y, ldir.x);
        float a = u_kind == 1 ? a0 + (2.0 * u - 1.0) * u_cone : 6.2831853 * u;
        dir = vec2(cos(a), sin(a));
        o = u_light_pos;
        g_spacing = (u_kind == 1 ? 2.0 * u_cone : 6.2831853) / float(u_rays);
        // a light inside glass / fluid: the march starts inside that medium (first event: the exit refraction)
        if (!off_view(o)) {
            uint id = mat_id(pix(o));
            if (id != 0u && see_through(kind_of(id))) g_start_medium = id;
        }
    }
    g_start = o;
    g_t0 = 0.0;
    march(o, dir, max_t, 0u);
}
