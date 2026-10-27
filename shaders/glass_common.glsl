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

#ifdef USE_SDF
vec2 sdf_grad(vec2 q, float h) {
    return vec2(sdf_at(u_sdf, q + vec2(h, 0.0), u_period) - sdf_at(u_sdf, q - vec2(h, 0.0), u_period),
                sdf_at(u_sdf, q + vec2(0.0, h), u_period) - sdf_at(u_sdf, q - vec2(0.0, h), u_period));
}
#endif

#ifdef RIM_SMOOTH_PX
// A smooth outward boundary normal for the rim pixel c of a body (caustics_normals.frag stores it per pixel for the
// caustics march, u_rim). The sdf is the distance to the rasterised outline: its gradient at the boundary follows the
// pixel stairs, up to ~20 degrees off the real edge, which fans refracted light into streaks. The G-buffer bevel
// normal of the body's rim pixels is the polygon edge's own perpendicular (analytic, F's world_fill), but the outline
// drawn over the rim carries a flat normal: from taps across +-RIM_SMOOTH_PX along the tangent, look inward up to
// RIM_DEPTH px for the first pixel of this body with a bevel normal, and average those (tent weights), which also
// blends the polygon's facets into a smooth curve. Taps more than RIM_CORNER (a cosine) off the centre tap's normal
// are past a sharp corner (a lens tip, a box corner): left out, so corners stay sharp. vec2(0) if c has none.
vec2 rim_normal(ivec2 c) {
    uint body = mat_id(c);
    vec2 n0 = bevel_n(c);  // only the direction to search inward along
#ifdef USE_SDF
    if (dot(n0, n0) <= 0.01) n0 = sdf_grad(vec2(c) + 0.5, 1.5);  // outline pixels carry a flat normal
#endif
    n0 = dot(n0, n0) > 1e-6 ? normalize(n0) : edge_normal(c, vec2(0.0));
    if (body == 0u || dot(n0, n0) < 0.5) return vec2(0.0);
    vec2 tg = vec2(-n0.y, n0.x), q = vec2(c) + 0.5;
    vec2 acc = vec2(0.0), centre = vec2(0.0);
    for (int j = 0; j <= 2 * RIM_TAPS; j++) {
        int i = (j + 1) / 2 * ((j & 1) == 1 ? 1 : -1);  // 0, 1, -1, 2, -2, ...: the centre tap first
        float f = float(i) / float(RIM_TAPS);
        vec2 base = q + tg * (f * RIM_SMOOTH_PX);
        for (int k = -1; k <= RIM_DEPTH; k++) {
            ivec2 d = pix(base - n0 * float(k));
            vec2 b = bevel_n(d);
            if (mat_id(d) == body && dot(b, b) > 0.01) {
                b = normalize(b);
                if (j == 0) centre = b;
                if (j == 0 || dot(b, centre) > RIM_CORNER) acc += (1.0 - 0.5 * abs(f)) * b;
                break;
            }
        }
        if (j == 0 && dot(centre, centre) == 0.0) return vec2(0.0);
    }
    return normalize(acc);
}
#endif

// caustics_normals.frag writes vec2(RIM_THIN_MARK, 0) on glass / fluid less than 2 px thick (a sliver, a zero-area
// body drawn as its outline): it has no usable boundary normal, so a ray crossing it goes straight on (hit_normal
// returns the caller's fallback, the ray's own direction, which refracts to itself) instead of fanning into streaks.
#define RIM_THIN_MARK 4.0
#ifdef USE_RIM_TEX
uniform sampler2D u_rim;  // rim_normal per pixel (caustics_normals.frag), 0 where there is none
#endif

// The outward normal of the boundary a ray crossed between `inner` (the body pixel) and `air` (the point just outside
// it): the smoothed rim normal when the includer has u_rim (USE_RIM_TEX); else with the SDF its gradient taken on the
// air side (smooth-ish, and right on outlines, which carry a flat G-buffer normal); else edge_normal of the body pixel.
vec2 hit_normal(vec2 inner, vec2 air, vec2 fallback) {
#ifdef USE_RIM_TEX
    // bilinear over the rim texels around `inner` that have a normal (a per-pixel constant would bunch the rays
    // crossing one pixel into fine streaks)
    vec2 g = inner - 0.5, fl = floor(g), fr = g - fl;
    vec2 rn = vec2(0.0);
    for (int k = 0; k < 4; k++) {
        ivec2 o = ivec2(k & 1, k >> 1);
        vec2 n = texelFetch(u_rim, pixi(ivec2(fl) + o), 0).rg;
        if (n.x > 0.5 * RIM_THIN_MARK) return fallback;  // degenerate glass: no refraction
        float w = (o.x == 1 ? fr.x : 1.0 - fr.x) * (o.y == 1 ? fr.y : 1.0 - fr.y);
        if (dot(n, n) > 0.25) rn += w * n;
    }
    if (dot(rn, rn) > 1e-6) return normalize(rn);
    rn = texelFetch(u_rim, pix(inner), 0).rg;
    if (rn.x > 0.5 * RIM_THIN_MARK) return fallback;
    if (dot(rn, rn) > 0.25) return normalize(rn);
#endif
#ifdef USE_SDF
    vec2 sg = sdf_grad(air, 2.0);
    if (dot(sg, sg) > 1e-4) return normalize(sg);
#endif
    return edge_normal(pix(inner), fallback);
}

// ---------------------------------------------------------------- in-plane ray march
// MARCH_HOOK(at, r, inside, t), if the includer defines it, sees the path: it is called with the ray's state for the
// straight piece that ends at `at` (a boundary point where an interaction is about to change r, or the final r.pos
// when the march returns), `inside` = that piece ran inside a medium, t = the path length there. The caustics
// splat each air piece as a line from these calls.
#ifdef CAUSTIC_PIECES  // caustics_rays.geom defines caustic_piece
void caustic_piece(vec2 at, vec3 thr, int events, int refr, bool inside, float t);
#define MARCH_HOOK(at, r, inside, t) caustic_piece(at, r.thr, r.events, r.refr, inside, t)
// Caustic rays get a bigger interaction budget than view rays: a stack of k lenses takes 2k refractions (in and out),
// and a ray that runs out stops there, leaving everything below the stack dark. The view / resolve programs keep
// MARCH_BOUNCES (their output is unchanged).
#define MARCH_BUDGET 12
// A light inside a glass / fluid body (caustics_rays.geom sets this to the body's id before march()): the march
// starts inside that medium, so the ray's first event is the exit refraction out of it.
uint g_start_medium = 0u;
// Extra march steps for this ray on top of MARCH_STEPS (caustics_rays.geom: a periodic sun's warm-up lap).
int g_extra_steps = 0;
#define MARCH_LOOP_STEPS (MARCH_STEPS + g_extra_steps)
#else
#define MARCH_BUDGET MARCH_BOUNCES
#define MARCH_LOOP_STEPS MARCH_STEPS
#endif
#ifndef MARCH_HOOK
#define MARCH_HOOK(at, r, inside, t)
#endif
// Light / view rays travelling in the plane: air (background, id 0) -> sdf steps; glass / fluid -> Snell in, march
// inside with Beer-Lambert absorption toward the body's colour, Snell out or total internal reflection; mirror ->
// reflect (tinted); opaque body -> stop. MARCH_BUDGET interactions at most.
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
#ifdef CAUSTIC_PIECES
    if (g_start_medium != 0u && see_through(kind_of(g_start_medium))) {
        inside = true;
        ior = max(g_mat.ior, 1.0);
        absorption = g_mat.absorption;
        body_col = texelFetch(u_albedo, pix(r.pos), 0).rgb;
        r.thr *= g_mat.transparency;
    }
#endif
    vec2 prev = r.pos;
    for (int i = 0; i < MARCH_LOOP_STEPS; i++) {
        if (t >= max_t || off_view(r.pos)) { r.end = inside ? 2 : 0; MARCH_HOOK(r.pos, r, inside, t); return r; }
        ivec2 c = pix(r.pos);
        uint id = mat_id(c);
        int k = kind_of(id);
#ifdef CAUSTIC_PIECES
        // an opaque body that casts no shadow (flag bit1 off: the world's walls) doesn't stop light in the light
        // pass, so it doesn't stop caustic light either (else the walls round the view block every directional ray)
        if (k == KIND_OPAQUE && (int(texelFetch(u_normal, c, 0).a * 255.0 + 0.5) & 2) == 0) k = -1;
#endif
        float step;
        if (!inside) {
            if (k == KIND_OPAQUE) {
                vec4 e = refine(prev, r.pos, 0);
                r.pos = e.zw; r.end = 1; MARCH_HOOK(r.pos, r, false, t); return r;
            }
            if (k >= 0) {  // mirror, glass or fluid: an interaction
                vec4 e = refine(prev, r.pos, 0);
                if (r.events >= MARCH_BUDGET) { r.end = 2; MARCH_HOOK(e.xy, r, false, t); return r; }
                MARCH_HOOK(e.xy, r, false, t);
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
                MARCH_HOOK(e.xy, r, true, t);
                vec2 n = hit_normal(e.xy, e.zw, r.dir);  // outward
                if (dot(n, r.dir) < 0.0) n = -n;
                vec2 rd = refract(r.dir, -n, ior);
                if (dot(rd, rd) < 1e-6) {  // total internal reflection
                    if (r.events >= MARCH_BUDGET) { r.pos = e.xy; r.end = 2; return r; }
                    r.events++;
                    r.dir = reflect(r.dir, -n);
                    r.pos = e.xy + r.dir;
                } else {
#ifdef CAUSTIC_PIECES  // out of budget: stop inside, like the entry branch (view rays keep their old exit rule)
                    if (r.events >= MARCH_BUDGET) { r.pos = e.xy; r.end = 2; return r; }
#endif
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
    MARCH_HOOK(r.pos, r, inside, t);
    return r;
}
