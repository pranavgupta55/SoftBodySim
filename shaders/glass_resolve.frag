// Glass / fluid / mirror resolve (gl_glass.GlassEffect, order 400): rewrites the lit scene's pixels of see-through
// and mirror bodies (and the fluid filter's bridges between fluid bodies); every other pixel is discarded, so it
// keeps the composed colour byte for byte.
//
// Pseudo-3D: the viewer looks straight down; a body is a bump whose surface normal is (bevel n.xy, sqrt(1 - |n|^2)).
//   refraction  what is behind the body, displaced by the bent view ray: sample "behind" at
//               p - n.xy / n.z * (1 - 1 / ior) * (REFRACT_DEPTH_PX + thickness / 2), at least MIN_REFRACT_PX on
//               the tilted rim, attenuated by Beer-Lambert exp(-absorption * thickness) toward the body's colour
//               (thickness = THICK_PER_EDGE x the edge distance, capped at THICK_CAP_PX of it)
//   blend       mix(the body's own lit colour, refracted, transparency)
//   Fresnel     Schlick with F0 = max(reflectivity, ((ior - 1) / (ior + 1))^2), cos = n.z: the rim reflects
//   mirror      mix(own lit colour, reflection tinted by albedo x tint_strength, reflectivity)
//   reflection  the view ray VIEW3 (tilted in-plane, see gl_glass.VIEW_TILT) reflected about the 3D normal; its
//               in-plane part points where it looks, MIRROR_DEPTH_PX * |R.xy| / R.z px far. With USE_SDF the ray
//               is marched in the plane from the nearest edge (glass_common.march: bounces, Snell, TIR); without,
//               the scene is sampled at that end point.
//   glint       a fixed key light GLINT_DIR (Blinn-Phong, shininess from roughness) plus L's own spec texture.
// "Behind" a glass body isn't in scene_lit (the body covers it): u_behind (the unlit floor: background, grid,
// trails) x the light ratio inpainted from the surrounding floor, or without u_behind the surrounding scene
// colour inpainted (push-pull, glass_pp_*.frag).
out vec4 f_color;

uniform sampler2D u_scene;       // shared["scene_lit"]
uniform sampler2D u_behind;      // unlit floor under the bodies, if u_behind_on
uniform bool u_behind_on;
uniform sampler2D u_fill_ratio;  // push-pull results (quarter res, LINEAR)
uniform sampler2D u_fill_col;
uniform sampler2D u_field;       // fluid field (half res, LINEAR), if u_fluid_on
uniform sampler2D u_fluid_col;   // blurred fluid colour: rgb / a
uniform sampler2D u_fluid_smooth;  // the thresholded liquid, blurred again: normal (gradient) and thickness
uniform bool u_fluid_on;
uniform float u_fluid_t;         // threshold: field >= it is liquid
uniform sampler2D u_spec;        // shared["spec"], if u_spec_on
uniform bool u_spec_on;

const vec3 VIEW = normalize(VIEW3);

vec2 fill_uv(ivec2 c) {
    return (vec2(c) + 0.5) * 0.25 / vec2(textureSize(u_fill_ratio, 0));
}

// a half-res fluid texture at render px q (bilinear)
vec4 half_at(sampler2D t, vec2 q) {
    return texture(t, wrapq(q) * 0.5 / vec2(textureSize(t, 0)));
}

vec3 light_ratio(ivec2 c) {
    return texture(u_fill_ratio, fill_uv(c)).rgb;
}

// Absorption follows the edge distance only up to this many px: a big clear body isn't smoked in its middle.
const float THICK_CAP_PX = 8.0;
// Glints: Blinn-Phong lobes are scaled by min(1, (shin + 2) / (SPEC_REF_SHIN + 2)), so a rough lobe is wide and dim
// instead of wide and full strength (the same normalisation as light_accum.frag).
const float SPEC_REF_SHIN = 32.0;
// A reflected ray that never leaves its own body within this many px past its end shows only the blurred
// surroundings (exp(-SELF_LOOK_PX / MIRROR_SOFT_PX) < 1%).
const float SELF_LOOK_PX = 5.0 * MIRROR_SOFT_PX;
const int SELF_WALK_STEPS = 400;  // >= (MARCH_MAX_T + SELF_LOOK_PX) / 1 px

// The floor at c (u_behind x the inpainted light ratio), if there is a floor texture.
vec3 floor_lit(ivec2 c) {
    return max((texelFetch(u_behind, c, 0).rgb + RATIO_EPS) * light_ratio(c) - RATIO_EPS, 0.0);
}

// What a ray sees at q in the plane: the lit scene; under a see-through body the floor.
vec3 env(vec2 q) {
    ivec2 c = pix(q);
    if (!see_through(kind_of(mat_id(c)))) return texelFetch(u_scene, c, 0).rgb;
    if (u_behind_on) return floor_lit(c);
    return texture(u_fill_col, fill_uv(c)).rgb;
}

// The blurred surroundings at q (inpainted under see-through / mirror bodies): what a reflection that stays on its
// own body (a flat mirror / glass top reflecting its own area) or a rough one shows. Never darker than the lit floor
// there, so a mirror doesn't show its own cast shadow (inpainted from around it) as a black hole.
vec3 fill_env(vec2 q, float blur_px) {
    ivec2 c = pix(q);
    vec3 f = texture(u_fill_col, fill_uv(c)).rgb;
    if (blur_px > 0.5) {  // 4 more taps around: the quarter-res fill alone blurs only ~4 px
        f += texture(u_fill_col, fill_uv(pix(q + vec2(blur_px, 0.0)))).rgb;
        f += texture(u_fill_col, fill_uv(pix(q - vec2(blur_px, 0.0)))).rgb;
        f += texture(u_fill_col, fill_uv(pix(q + vec2(0.0, blur_px)))).rgb;
        f += texture(u_fill_col, fill_uv(pix(q - vec2(0.0, blur_px)))).rgb;
        f *= 0.2;
    }
    if (u_behind_on && mat_id(c) != 0u) f = max(f, floor_lit(c));
    return f;
}

vec3 normal3(vec2 n) {
    float l2 = dot(n, n);
    if (l2 > 0.97) { n *= sqrt(0.97 / l2); l2 = 0.97; }
    return vec3(n, sqrt(1.0 - l2));
}

// What the body `self_id` reflects at p. The ray starts on the body (at the nearest edge point on the tilted rim),
// walks out of it along dir in 1 px steps (longer where the edge is far) and bisects the exit to
// sub-pixel precision, so where the reflection starts showing the surroundings is a smooth line, not one quantised
// by SDF steps (a comb). Past the exit it marches on (glass_common.march: bounces, Snell, TIR). A ray that ends
// still on its own body shows the blurred surroundings, blending into what lies just past the body's edge ALONG
// the ray over MIRROR_SOFT_PX; one that just left blends from that toward its marched end over 2 x MIRROR_SOFT_PX:
// no hard switch between the two. Roughness blurs it all toward the surroundings at the ray's end.
vec3 reflection(ivec2 p, vec3 n3, uint self_id, float d, float roughness) {
    vec3 R = reflect(VIEW, n3);
    float l = length(R.xy);
    vec2 start = vec2(p) + 0.5;
    float nl = length(n3.xy);
    if (nl > 0.05) start += n3.xy / nl * d;  // the nearest edge point: the rim reflects what's beside it
    vec2 dir = l < 1e-4 ? edge_normal(pix(start), vec2(0.0, -1.0)) : R.xy / l;
    float travel = l < 1e-4 ? 0.0 : min(MIRROR_DEPTH_PX * l / max(R.z, 0.05), MARCH_MAX_T);
    float rw = smoothstep(0.05, 0.6, roughness);
    float blur = rw * 6.0;

    // walk out of the own body: the last point on it (a) and the first past it (b)
    vec2 a = start, b = start;
    float t = 0.0;
    bool left = mat_id(pix(start)) != self_id;
    for (int i = 0; i < SELF_WALK_STEPS && !left && t <= travel + SELF_LOOK_PX; i++) {
        a = b;
        // the edge is at least the G-buffer's edge distance away (from the pixel centre: - 1 for anywhere in it), so a
        // deep interior is crossed in a few steps and no exit is stepped over
        float s = max(1.0, edge_d(pix(b)) - 1.0);
        b += dir * s;
        t += s;
        left = mat_id(pix(b)) != self_id;
    }
    float t_exit = 0.0;
    if (left && t > 0.0) {
        for (int j = 0; j < 7; j++) {
            vec2 m = 0.5 * (a + b);
            if (mat_id(pix(m)) == self_id) a = m; else b = m;
        }
        t_exit = dot(b - start, dir);
    }
    vec2 end = start + dir * travel;
    if (!left) return fill_env(end, blur);  // stays on its own body well past its end
    // just past the body's edge, along the ray: a small footprint (3 across x 2 along, off the own body), else the
    // first pixel past a stair of a slanted edge decides a whole column of the fade (vertical streaks)
    vec2 side = vec2(-dir.y, dir.x);
    vec4 acc = vec4(0.0);
    for (int k = 0; k < 6; k++) {
        vec2 q = b + dir * (0.5 + float(k / 3)) + side * float(k % 3 - 1);
        if (mat_id(pix(q)) != self_id) acc += vec4(env(q), 1.0);
    }
    vec3 edge = acc.a > 0.0 ? acc.rgb / acc.a : env(b);
    float s = travel - t_exit;
    vec3 col;
    if (s <= 0.0) {
        col = mix(fill_env(end, 0.0), edge, exp(s / MIRROR_SOFT_PX));
    } else {
        vec3 far;
#ifdef USE_SDF
        Ray r = march(b, dir, s, 0u);
        far = (r.end == 1 ? texelFetch(u_scene, pix(r.pos), 0).rgb : env(r.pos)) * r.thr;
        end = r.pos;
#else
        far = env(end);
#endif
        col = mix(edge, far, smoothstep(0.0, 2.0 * MIRROR_SOFT_PX, s));
    }
    return mix(col, fill_env(end, blur), rw);
}

vec3 glint(vec3 n3, float specular, float roughness) {
    vec3 h = normalize(normalize(GLINT_DIR) + vec3(0.0, 0.0, 1.0));
    float shin = exp2(10.0 * (1.0 - clamp(roughness, 0.0, 1.0)));
    float norm = min(1.0, (shin + 2.0) / (SPEC_REF_SHIN + 2.0));
    return vec3(GLINT_STRENGTH * specular * norm * pow(max(dot(n3, h), 0.0), shin));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_clip.x || p.y >= u_clip.y) discard;
    uint id = mat_id(p);
    int kind = kind_of(id);
    Mat m = g_mat;
    vec4 nb = texelFetch(u_normal, p, 0);
    // a glass / mirror body's black border (filled view, FLAG_BORDER) keeps its composed colour, as unlit: the optics
    // stop at the body's fill (fluid borders melt into the liquid like the rest of it)
    if ((uint(nb.a * 255.0 + 0.5) & 8u) != 0u && kind != KIND_FLUID) discard;
    vec2 n = nb.rg * 2.0 - 1.0;
    float d = nb.b * 255.0;
    vec3 own = texelFetch(u_scene, p, 0).rgb;
    vec3 body = texelFetch(u_albedo, p, 0).rgb;  // the colour absorbed toward
    float thick = THICK_PER_EDGE * min(d, THICK_CAP_PX);

    if (u_fluid_on && (kind == KIND_FLUID || kind < 0)) {
        vec2 pc = vec2(p) + 0.5;
        float f = half_at(u_field, pc).r;
        if (f < u_fluid_t) {
            if (kind == KIND_FLUID) { f_color = vec4(env(vec2(p) + 0.5), 1.0); return; }  // filtered away
            discard;
        }
        // liquid: one smooth surface over the whole filtered field
        vec2 g = vec2(half_at(u_fluid_smooth, pc + vec2(1.0, 0.0)).r - half_at(u_fluid_smooth, pc - vec2(1.0, 0.0)).r,
                      half_at(u_fluid_smooth, pc + vec2(0.0, 1.0)).r - half_at(u_fluid_smooth, pc - vec2(0.0, 1.0)).r)
                 * 0.5;
        n = -g * FLUID_NORMAL_GAIN;
        thick = half_at(u_fluid_smooth, pc).r * FLUID_THICK_PX;
        if (kind != KIND_FLUID) {
            // a bridge pixel: take the material of a fluid body nearby (rings of 8 directions, growing)
            uint sid = 0u;
            for (int r = 1; r <= FLUID_SEARCH && sid == 0u; r += 2) {
                for (int k = 0; k < 8; k++) {
                    ivec2 o = ivec2(k == 0 || k == 1 || k == 7 ? r : (k == 3 || k == 4 || k == 5 ? -r : 0),
                                    k == 1 || k == 2 || k == 3 ? r : (k == 5 || k == 6 || k == 7 ? -r : 0));
                    uint cid = mat_id(pixi(p + o));
                    if (kind_of(cid) == KIND_FLUID) { sid = cid; break; }
                }
            }
            if (sid == 0u) discard;
            id = sid;
        }
        kind_of(id);
        m = g_mat;
        kind = KIND_FLUID;
        vec4 fc = half_at(u_fluid_col, pc);
        body = fc.a > 1e-3 ? fc.rgb / fc.a : texelFetch(u_albedo, p, 0).rgb;
        own = body * light_ratio(p);
    }

    if (see_through(kind)) {
        vec3 n3 = normal3(n);
        float ior = max(m.ior, 1.0);
        vec2 off = -n3.xy / max(n3.z, 0.3) * (1.0 - 1.0 / ior) * (REFRACT_DEPTH_PX + 0.5 * thick);
        float ol = length(off);
        if (ol > 1e-4 && ol < MIN_REFRACT_PX && length(n3.xy) > 0.05 && ior > 1.0) off *= MIN_REFRACT_PX / ol;
        vec3 refr = env(vec2(p) + 0.5 + off) * mix(body, vec3(1.0), exp(-m.absorption * thick));
        vec3 col = mix(own, refr, m.transparency);
        float f0 = max(m.reflectivity, pow((ior - 1.0) / (ior + 1.0), 2.0));
        float F = f0 + (1.0 - f0) * pow(1.0 - n3.z, 5.0);
        if (ior > 3.0) F = min(F, max(m.reflectivity, 0.5));  // a very high ior doesn't turn the rim into a mirror
        if (F > 0.01) col = mix(col, reflection(p, n3, id, d, m.roughness), F);
        if (u_spec_on) col += texelFetch(u_spec, p, 0).rgb * m.transparency * (1.0 - F);
        col += glint(n3, m.specular, m.roughness);
        f_color = vec4(col, 1.0);
        return;
    }
    if (kind == KIND_MIRROR) {
        vec3 n3 = normal3(n);
        vec3 refl = reflection(p, n3, id, d, m.roughness)
                    * mix(vec3(1.0), texelFetch(u_albedo, p, 0).rgb, m.tint_strength);
        vec3 col = mix(own, refl, m.reflectivity);
        if (u_spec_on) col += texelFetch(u_spec, p, 0).rgb * m.reflectivity;
        col += glint(n3, m.specular, m.roughness);
        f_color = vec4(col, 1.0);
        return;
    }
    discard;
}
