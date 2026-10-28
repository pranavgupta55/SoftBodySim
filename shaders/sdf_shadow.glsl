// Soft shadows from the shared signed distance field (gl_sdf.py's shared["sdf"]: R32F, signed distance in render px
// to the nearest shadow-casting occluder boundary, negative inside, minimum-image across periodic edges; LINEAR
// filtered). Self-contained: include it into any shader (it declares no uniforms).
//
// Coordinates are render px, y down, continuous: texel (x, y) covers [x, x + 1) x [y, y + 1), so a fragment's
// own position is gl_FragCoord.xy. `period` is the map's repeat in render px per axis, (0, 0) = not periodic.
//
// sdf_shadow(sdf, p, dir, max_t, softness_px, period) -> 1 lit .. 0 shadowed, for a light at p + dir * max_t (dir
// a unit vector toward the light; for a directional light pass the march length you want, e.g. the screen
// diagonal). softness_px is the light's radius: the penumbra's angular half-width is softness_px / max_t, the
// classic k = max_t / softness_px of Inigo Quilez's soft shadow (iquilezles.org/articles/rmshadows), with his
// improved estimator (the closest approach between two samples, from the two distance circles) and his centred
// penumbra, so a penumbra extends both sides of the hard shadow edge. Softness is floored at SDF_MIN_SOFT px so a
// hard light still gets an anti-aliased 1-2 px edge.
// A start point inside an occluder first walks out of that body along dir (sphere tracing on |d|): a body never
// shadows itself (its faces are shaded by the lights' normals instead) but still shadows everything behind it.
// Bodies that touch merge in the field, so a pixel's walk out continues through a body pressed against it.
//
// Variants (same march, sdf_shadow keeps its behaviour for every other caller):
// - sdf_shadow_pt(...): point / spot lights. A light embedded in an occluder (deeper than ~SDF_LIGHT_INSIDE) walks
//   out of it toward the pixel first (the light's own body never shadows it, mirroring the pixel rule); a light on
//   or just under a surface moves out a pixel past it instead, so the body still shadows its far side. A light
//   disk that doesn't fit in its free space is clipped to it near the light (sdf_shadow_r), so a light touching a
//   body lights its side evenly instead of dimming the room with a quantised fan.
// - sdf_shadow_dir(..., fade_start): directional lights; fade_start > 0 fades the shadow out between fade_start and
//   max_t along the march (periodic maps: a wrapped shadow ends softly instead of in a cut at the march length).
// - sdf_shadow_tr(sdf_tr, tint, p, dir, max_t, softness_px, period, point, fade_start, scale) -> vec3: the same
//   march over a field of translucent bodies only (gl_sdf's shared["sdf_tr"]), returning the light let through per
//   channel. tint (shared["sdf_tint"]) holds each texel's nearest translucent body's transmittance T (x scale).
//   Each sample's soft visibility v (the estimator's, as above) becomes mix(T, 1, v) and the result is the minimum
//   over the samples: a ray through one body keeps T of it with sdf_shadow's soft edge, a ray through several the
//   least T among them (the minimum over crossings, not a product, so a row of clear bodies doesn't go dark).
//   Opaque occluders are another field's job: multiply with sdf_shadow_light over gl_sdf's shared["sdf_opaque"].
#ifndef SDF_SHADOW_GLSL
#define SDF_SHADOW_GLSL

const int SDF_EXIT_STEPS = 16;   // walk-out steps from inside a body
const int SDF_MARCH_STEPS = 96;   // sphere-tracing steps toward the light
const int SDF_FINISH_STEPS = 8;   // uniform samples over what the march didn't reach
const float SDF_MIN_STEP = 1.0;  // px
const float SDF_INSIDE_STEP = 0.5;  // px, march step while inside an occluder
const float SDF_OWN_MARGIN = 1.0;  // px: see sdf_shadow's own-body phase
const float SDF_MIN_SOFT = 0.75;  // px: softness floor (anti-aliased hard shadows)
const float SDF_THIN = 0.5;  // px every occluder counts fatter (thin ropes' shadows stay continuous)
const float SDF_LIGHT_INSIDE = 3.0;  // px: sdf_shadow_pt, a light this deep in a body starts to skip it

// The field at p (bilinear). Off a non-periodic texture, a lower bound from the nearest texel centre (occluders
// all lie on the texture: sqrt(d^2 + offset^2) for outside points).
float sdf_at(sampler2D sdf, vec2 p, vec2 period) {
    vec2 size = vec2(textureSize(sdf, 0));
    if (period.x > 0.0) p.x -= period.x * floor(p.x / period.x);
    if (period.y > 0.0) p.y -= period.y * floor(p.y / period.y);
    vec2 c = clamp(p, vec2(0.5), size - 0.5);
    float d = texture(sdf, c / size).r;
    vec2 off = p - c;
    float o2 = dot(off, off);
    return (o2 > 0.0 && d > 0.0) ? sqrt(d * d + o2) : d;
}

// The estimator's k * h / s for a sample h at s (rem: the march length, so rem - s is the distance to the light).
// - Every occluder counts SDF_THIN px fatter, ramping in over the first / last 5 px of the march so contact pixels and
//   the light's own surroundings stay lit: a 1 px rope's field only reaches 0 where a ray crosses its raster stairs
//   at a corner and -0.5 elsewhere, which striped its shadow.
// - clear > 0 (point lights): the light's free space, the field at the light. A light disk that doesn't fit there
//   is clipped to it, near the light: within 2 x softness of the light the penumbra narrows from the light's
//   softness to `clear`, so a light touching a body doesn't half-shadow the whole room through its own disk, while
//   a ray through a wall next to the light is still blocked (h < 0 stays negative).
float sdf_shadow_r(float h, float s, float rem, float k, float soft, float clear) {
    float sl = rem - s;
    float he = h - min(SDF_THIN, 0.1 * min(s, sl));
    if (clear > 0.0 && clear < soft) {
        k *= soft / mix(clear, soft, smoothstep(0.0, 2.0 * soft, sl));
    }
    return k * he / max(s, 1e-3);
}

// The estimator's minimum r (clamped to [-1, 1]) as a visibility: 0 shadowed .. 1 lit, smooth at both ends.
float sdf_shadow_vis(float r) {
    r = clamp(r, -1.0, 1.0);
    return 0.25 * (1.0 + r) * (1.0 + r) * (2.0 - r);
}

// The tint texture (sdf_shadow_tr) at p, wrapped like sdf_at.
vec3 sdf_tint_at(sampler2D tint, vec2 p, vec2 period) {
    vec2 size = vec2(textureSize(tint, 0));
    if (period.x > 0.0) p.x -= period.x * floor(p.x / period.x);
    if (period.y > 0.0) p.y -= period.y * floor(p.y / period.y);
    return texture(tint, clamp(p, vec2(0.5), size - 0.5) / size).rgb;
}

// The march shared by the variants. light_end: the light's disk is clipped to its free space (point lights);
// walk_light: march only to where the light's body ends toward the pixel; fade_start > 0: the directional fade.
// tr (sdf_shadow_tr): sdf is a translucent field and tint its transmittance (x tr_scale); the result is per channel.
// Otherwise tint is unused (pass sdf) and all three channels are the visibility.
vec3 sdf_shadow_march(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period,
                      bool light_end, bool walk_light, float fade_start, sampler2D tint, bool tr, float tr_scale) {
    float t = 0.0;
    float d = sdf_at(sdf, p, period);
    for (int i = 0; i < SDF_EXIT_STEPS && d < 0.0; i++) {  // leave the body p is in
        t += max(-d, 0.5);
        if (t >= max_t) return vec3(1.0);
        d = sdf_at(sdf, p + dir * t, period);
    }
    if (d < 0.0) return vec3(1.0);  // still inside one long body: its own, so lit
    float end_t = max_t;
    if (walk_light) {  // a light inside an occluder: walk out of it toward the pixel, march only to there
        vec2 lp = p + dir * max_t;
        float tl = 0.0;
        float dl = sdf_at(sdf, lp, period);
        for (int i = 0; i < SDF_EXIT_STEPS && dl < 0.0; i++) {
            tl += max(-dl, 0.5);
            dl = sdf_at(sdf, lp - dir * tl, period);
        }
        if (tl > 0.0) end_t = max_t - tl - 1.0;  // end a pixel outside the light's body
    }
    vec2 o = p + dir * t;
    float rem = end_t - t;
    if (rem <= 0.5) return vec3(1.0);  // the light is right here (or the pixel and the light share a body)
    float soft = max(softness_px, SDF_MIN_SOFT);
    float k = max_t / soft;
    // light_end: the light's free space (sdf_shadow_pt has moved it at least ~1 px off any surface)
    float clear = light_end ? max(sdf_at(sdf, o + dir * rem, period), SDF_MIN_SOFT) : 0.0;
    bool fade = fade_start > 0.0 && fade_start < max_t;
    float last = rem - 0.5;  // the final sample, always taken
    float res = 1.0;
    vec3 res_tr = vec3(1.0);  // tr: the light let through so far
    float r_last = 1.0;  // tr: this sample's r (deep in a body's umbra the march may stride)
    float s = min(SDF_MIN_STEP, last);
    float ps = 0.0, ph = -1.0;  // the previous sample (ph <= 0: none usable)
    // After walking out, the ray first moves away from its own body (h rising, slowly when it left at a grazing
    // angle near the terminator): that stretch doesn't count, or the body would speckle-shadow its own lit face.
    // Counting starts once h falls SDF_OWN_MARGIN below the most it has been (something else ahead, or a concave
    // part of the same body); smaller dips are the rasterised outline's stairs, which a grazing ray re-enters.
    bool counting = t == 0.0;
    float hmax = d;
    for (int i = 0; i < SDF_MARCH_STEPS + SDF_FINISH_STEPS; i++) {
        // past SDF_MARCH_STEPS the march ran out before the light: finish with uniform samples, never assume lit
        bool finishing = i >= SDF_MARCH_STEPS - 1;
        float h = sdf_at(sdf, o + dir * s, period);
        hmax = max(hmax, h);
        counting = counting || h < hmax - SDF_OWN_MARGIN;
        if (counting) {
            float r = sdf_shadow_r(h, s, rem, k, soft, clear);  // h < 0 (inside): the plain estimator
            if (h > 0.0) {
                // improved estimator: the two distance circles (previous sample, this one) meet at the ray's
                // closest possible approach between them, y back from s at height dd. When the circles don't
                // overlap (a step longer than ph + h, e.g. over a 1 px rope whose field never goes negative) an
                // occluder may sit in the gap: count it as touching the ray there.
                float dl = s - ps;
                if (ph > 0.0 && dl > 1e-3) {
                    float y = -1.0, dd = 0.0;
                    if (dl >= ph + h) {
                        // only a sub-pixel gap (bigger ones are the step floor's, which accepts them)
                        if (dl - ph - h < SDF_MIN_STEP) y = dl * h / (ph + h);
                    } else {
                        y = (h * h - ph * ph + dl * dl) / (2.0 * dl);
                        dd = sqrt(max(h * h - y * y, 0.0));
                    }
                    if (y > 0.0 && y < dl) {
                        float sy = s - y;
                        r = min(r, sdf_shadow_r(dd, sy, rem, k, soft, clear));
                    }
                }
            }
            // the fade: occluders past fade_start count less and less, none at max_t. Added rather than
            // max(r, fade): the minimum of max() sits on a sharp crossing whose sampled value jumped between
            // neighbouring rays (contour arcs); a smooth sum keeps the minimum smooth.
            if (fade) r = max(r, -1.0) + 2.0 * smoothstep(fade_start, max_t, t + s);
            if (tr) {
                // a translucent body's shadow is its transmittance where it would be black
                r_last = r;
                if (r < 1.0) {
                    vec3 tt = clamp(sdf_tint_at(tint, o + dir * s, period) * tr_scale, 0.0, 1.0);
                    res_tr = min(res_tr, mix(tt, vec3(1.0), sdf_shadow_vis(r)));
                    if (max(res_tr.r, max(res_tr.g, res_tr.b)) <= 0.002) break;
                }
            } else {
                res = min(res, r);
                if (res <= -1.0) break;
            }
        }
        if (s >= last) break;
        ps = s;
        ph = h;
        float step;
        if (finishing) {
            step = (last - s) / float(max(SDF_MARCH_STEPS + SDF_FINISH_STEPS - 2 - i, 1));
        } else if (tr && h < 0.0 && r_last <= -1.0) {
            // tr, deep in a body's umbra (what it lets through is counted): stride through it toward its far side
            step = max(-h, SDF_INSIDE_STEP);
        } else if (h < 0.0) {
            // Inside an occluder (a penumbra's dark side, the ray clipping a corner) step a fixed SDF_INSIDE_STEP:
            // stepping |h| there makes the sampled minimum of h jump as the step positions slide from pixel to
            // pixel, which drew wavy contour bands across penumbras. Deep inside, res hits -1 and breaks.
            step = SDF_INSIDE_STEP;
        } else {
            // sphere tracing with a floor that grows with the penumbra's width at s, but never more than a pixel
            // past h (a longer step could jump a thin wall, and whether it lands in it drew contour bands); close
            // to a surface step under h so a gap of a pixel or two between merged shadows isn't stepped over
            step = max(h, min(max(SDF_MIN_STEP, 0.25 * s / k), h + SDF_MIN_STEP));
            if (h < 2.0) step = min(step, max(0.9 * h, SDF_INSIDE_STEP));
        }
        s = min(s + step, last);  // the last step lands on the final sample
    }
    return tr ? res_tr : vec3(sdf_shadow_vis(res));
}

float sdf_shadow(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period) {
    return sdf_shadow_march(sdf, p, dir, max_t, softness_px, period, false, false, 0.0, sdf, false, 1.0).x;
}

// A light's shadow: point or directional (with its fade), over an opaque field (sdf_shadow_light) or a translucent
// one (sdf_shadow_tr). One march call site each (the march inlines, and every extra copy costs registers and
// occupancy).
vec3 sdf_shadow_light_c(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period, bool point,
                        float fade_start, sampler2D tint, bool tr, float tr_scale) {
    vec2 v = dir * max_t;  // to the light, or to where it's moved out of the surface it's on
    bool walk = false;
    if (point) {
        vec2 lp = p + v;
        float dl = sdf_at(sdf, lp, period);
        if (dl < 1.0) {
            // On or inside an occluder. Near the surface (touching it, an outline, a light dragged onto a body)
            // the light moves out to a pixel past the nearest surface, so the body still shadows its far side;
            // deeper than SDF_LIGHT_INSIDE the light's own body is skipped entirely and the light reaches
            // everything outside it.
            walk = -dl > SDF_LIGHT_INSIDE;
            if (!walk) {
                vec2 g = vec2(sdf_at(sdf, lp + vec2(1.0, 0.0), period) - sdf_at(sdf, lp - vec2(1.0, 0.0), period),
                              sdf_at(sdf, lp + vec2(0.0, 1.0), period) - sdf_at(sdf, lp - vec2(0.0, 1.0), period));
                float gl = length(g);
                walk = gl <= 1e-4;  // no direction out (a thin body's middle): walk out toward the pixel
                if (!walk) {
                    v = lp + g / gl * (1.0 - dl) - p;
                }
            }
        }
    }
    float lv = length(v);
    return lv > 0.5 ? sdf_shadow_march(sdf, p, v / lv, lv, softness_px, period, point, walk,
                                       point ? 0.0 : fade_start, tint, tr, tr_scale) : vec3(1.0);
}

float sdf_shadow_light(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period, bool point,
                       float fade_start) {
    return sdf_shadow_light_c(sdf, p, dir, max_t, softness_px, period, point, fade_start, sdf, false, 1.0).x;
}

vec3 sdf_shadow_tr(sampler2D sdf_tr, sampler2D tint, vec2 p, vec2 dir, float max_t, float softness_px,
                   vec2 period, bool point, float fade_start, float scale) {
    return sdf_shadow_light_c(sdf_tr, p, dir, max_t, softness_px, period, point, fade_start, tint, true, scale);
}

float sdf_shadow_pt(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period) {
    return sdf_shadow_light(sdf, p, dir, max_t, softness_px, period, true, 0.0);
}

float sdf_shadow_dir(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period,
                     float fade_start) {
    return sdf_shadow_light(sdf, p, dir, max_t, softness_px, period, false, fade_start);
}

#endif
