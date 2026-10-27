// Shared by gl_glass.py's resolve, fluid and caustics passes (prepended after #version, the material include
// (mat_fetch) and, with USE_SDF, the sdf_at include). Pixels are render px, y down, texel (x, y) = pixel (x, y).
// Constants (KIND_*, MARCH_*, STEP_*) come as #defines from gl_glass.py.
//
// G-buffer conventions used here (lighting contract): normal.rg = n.xy * 0.5 + 0.5, the bevel normal's in-plane part,
// pointing OUT of the body (away from its interior, toward the nearest edge); normal.b = distance to that edge / 255.

uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform usampler2D u_mat;
uniform sampler2D u_table;   // material table (mat_fetch)
uniform ivec2 u_clip;        // world view size: pixels at or past it are off the view
uniform vec2 u_period;       // periodic map size in render px per axis (0 = that axis doesn't wrap)
#ifdef USE_SDF
uniform sampler2D u_sdf;
#endif

bool off_view(vec2 q) {
    return (u_period.x <= 0.0 && (q.x < 0.0 || q.x >= float(u_clip.x)))
        || (u_period.y <= 0.0 && (q.y < 0.0 || q.y >= float(u_clip.y)));
}

vec2 wrapq(vec2 q) {
    q.x = u_period.x > 0.0 ? mod(q.x, u_period.x) : clamp(q.x, 0.0, float(u_clip.x) - 0.5);
    q.y = u_period.y > 0.0 ? mod(q.y, u_period.y) : clamp(q.y, 0.0, float(u_clip.y) - 0.5);
    return q;
}

ivec2 pix(vec2 q) {
    return clamp(ivec2(floor(wrapq(q))), ivec2(0), u_clip - 1);
}

ivec2 pixi(ivec2 c) {
    return pix(vec2(c) + 0.5);
}

uint mat_id(ivec2 c) {
    return texelFetch(u_mat, c, 0).r;
}

// One-entry cache: the march and the fluid search fetch the same few ids over and over.
uint g_id = 0xffffffffu;
Mat g_mat;
int g_kind = -1;

int kind_of(uint id) {  // -1 = background (id 0)
    if (id == 0u) return -1;
    if (id != g_id) {
        g_id = id;
        g_mat = mat_fetch(u_table, id);
        g_kind = int(float(g_mat.kind) + 0.5);
    }
    return g_kind;
}

bool see_through(int kind) {
    return kind == KIND_GLASS || kind == KIND_FLUID;
}

float edge_d(ivec2 c) {
    return texelFetch(u_normal, c, 0).b * 255.0;
}

vec2 bevel_n(ivec2 c) {
    return texelFetch(u_normal, c, 0).rg * 2.0 - 1.0;
}

// The in-plane outward boundary normal at a body pixel: the bevel normal's direction, or (deep inside the flat
// interior, where the bevel normal is 0) minus the gradient of the edge distance. `fallback` if both vanish.
vec2 edge_normal(ivec2 c, vec2 fallback) {
    vec2 n = bevel_n(c);
    if (dot(n, n) > 0.01) return normalize(n);
    vec2 g = vec2(edge_d(pixi(c + ivec2(1, 0))) - edge_d(pixi(c - ivec2(1, 0))),
                  edge_d(pixi(c + ivec2(0, 1))) - edge_d(pixi(c - ivec2(0, 1))));
    if (dot(g, g) > 1e-6) return -normalize(g);
    return fallback;
}

// The outward normal of the boundary a ray crossed between `inner` (the body pixel) and `air` (the point just outside
// it): with the SDF its gradient taken on the air side (smooth, and right on outlines, which carry a flat G-buffer
// normal); else edge_normal of the body pixel.
vec2 hit_normal(vec2 inner, vec2 air, vec2 fallback) {
#ifdef USE_SDF
    vec2 sg = vec2(sdf_at(u_sdf, air + vec2(2.0, 0.0), u_period) - sdf_at(u_sdf, air - vec2(2.0, 0.0), u_period),
                   sdf_at(u_sdf, air + vec2(0.0, 2.0), u_period) - sdf_at(u_sdf, air - vec2(0.0, 2.0), u_period));
    if (dot(sg, sg) > 1e-4) return normalize(sg);
#endif
    return edge_normal(pix(inner), fallback);
}

// ---------------------------------------------------------------- in-plane ray march
// Light / view rays travelling in the plane: air (background, id 0) -> sdf steps; glass / fluid -> Snell in, march
// inside with Beer-Lambert absorption toward the body's colour, Snell out or total internal reflection; mirror ->
// reflect (tinted); opaque body -> stop. MARCH_BOUNCES interactions at most.
struct Ray {
    vec2 pos;    // where it stopped (unwrapped px)
    vec2 dir;
    vec3 thr;    // throughput
    int events;  // reflections / refractions so far
    int refr;    // of which refractions into or out of glass / fluid
    int end;     // 0: ran its whole length (or left the view) in air, 1: hit an opaque body at pos, 2: stopped
                 // inside a medium or out of bounces (pos is the last point), 3: out of steps
};

float air_step(vec2 q) {
#ifdef USE_SDF
    return clamp(sdf_at(u_sdf, q, u_period), STEP_MIN_PX, STEP_MAX_PX);
#else
    return STEP_NOSDF_PX;
#endif
}

float medium_step(vec2 q) {
#ifdef USE_SDF
    float s = sdf_at(u_sdf, q, u_period);  // negative inside occluders: |s| = distance to leave (if glass occludes)
    return s < 0.0 ? clamp(-s, STEP_MIN_PX, STEP_MAX_PX) : STEP_MEDIUM_PX;
#else
    return STEP_MEDIUM_PX;
#endif
}

// state of a point for the boundary search: 0 air, 1 inside a see-through medium, 2 something else
int medium_state(vec2 q) {
    int k = kind_of(mat_id(pix(q)));
    return k < 0 ? 0 : (see_through(k) ? 1 : 2);
}

// Bisect the boundary between a (state sa) and b (another state): returns the last point still in state sa in .xy
// and the first point past it in .zw.
vec4 refine(vec2 a, vec2 b, int sa) {
    for (int j = 0; j < 4; j++) {
        vec2 m = 0.5 * (a + b);
        if (medium_state(m) == sa) a = m; else b = m;
    }
    return vec4(a, b);
}

Ray march(vec2 o, vec2 dir, float max_t, uint self_id) {
    Ray r = Ray(o, normalize(dir), vec3(1.0), 0, 0, 0);
    float t = 0.0;
    // leave the body the ray starts from (a view ray starting on a mirror / glass pixel)
    if (self_id != 0u) {
        for (int i = 0; i < MARCH_STEPS && t < max_t; i++) {
            if (mat_id(pix(r.pos)) != self_id) break;
            float s = medium_step(r.pos);
            r.pos += r.dir * s;
            t += s;
        }
    }
    bool inside = false;
    float ior = 1.0, absorption = 0.0;
    vec3 body_col = vec3(1.0);
    vec2 prev = r.pos;
    for (int i = 0; i < MARCH_STEPS; i++) {
        if (t >= max_t || off_view(r.pos)) { r.end = inside ? 2 : 0; return r; }
        ivec2 c = pix(r.pos);
        uint id = mat_id(c);
        int k = kind_of(id);
        float step;
        if (!inside) {
            if (k == KIND_OPAQUE) {
                vec4 e = refine(prev, r.pos, 0);
                r.pos = e.zw; r.end = 1; return r;
            }
            if (k >= 0) {  // mirror, glass or fluid: an interaction
                if (r.events >= MARCH_BOUNCES) { r.end = 2; return r; }
                vec4 e = refine(prev, r.pos, 0);
                ivec2 hc = pix(e.zw);
                kind_of(mat_id(hc));
                Mat m = g_mat;
                vec2 n = hit_normal(e.zw, e.xy, -r.dir);
                if (dot(n, r.dir) > 0.0) n = -n;
                r.events++;
                if (k == KIND_MIRROR) {
                    r.dir = reflect(r.dir, n);
                    r.thr *= m.reflectivity * mix(vec3(1.0), texelFetch(u_albedo, hc, 0).rgb, m.tint_strength);
                    r.pos = e.xy + r.dir;
                } else {
                    vec2 rd = refract(r.dir, n, 1.0 / max(m.ior, 1.0));
                    if (dot(rd, rd) < 1e-6) {
                        r.dir = reflect(r.dir, n);
                        r.pos = e.xy + r.dir;
                    } else {
                        r.dir = normalize(rd);
                        r.refr++;
                        inside = true;
                        ior = max(m.ior, 1.0);
                        absorption = m.absorption;
                        body_col = texelFetch(u_albedo, hc, 0).rgb;
                        r.thr *= m.transparency;
                        r.pos = e.zw + r.dir;
                    }
                }
                prev = r.pos;
                t += 1.0;
                continue;
            }
            step = air_step(r.pos);
        } else {
            if (k >= 0 && see_through(k)) {
                step = medium_step(r.pos);
                r.thr *= mix(body_col, vec3(1.0), exp(-absorption * step));
            } else {  // left the medium between prev and pos
                vec4 e = refine(prev, r.pos, 1);
                ivec2 ic = pix(e.xy);
                vec2 n = hit_normal(e.xy, e.zw, r.dir);  // outward
                if (dot(n, r.dir) < 0.0) n = -n;
                vec2 rd = refract(r.dir, -n, ior);
                if (dot(rd, rd) < 1e-6) {  // total internal reflection
                    if (r.events >= MARCH_BOUNCES) { r.pos = e.xy; r.end = 2; return r; }
                    r.events++;
                    r.dir = reflect(r.dir, -n);
                    r.pos = e.xy + r.dir;
                } else {
                    r.events++;
                    r.refr++;
                    r.dir = normalize(rd);
                    inside = false;
                    r.pos = e.zw + r.dir;
                }
                prev = r.pos;
                t += 1.0;
                continue;
            }
        }
        step = min(step, max_t - t);  // land exactly at max_t (else the end points bunch up on step boundaries)
        prev = r.pos;
        r.pos += r.dir * step;
        t += step;
    }
    r.end = 3;  // out of steps: pos is short of the length asked for
    return r;
}
