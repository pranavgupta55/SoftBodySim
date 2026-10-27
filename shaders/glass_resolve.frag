// Glass / fluid / mirror resolve (gl_glass.GlassEffect, order 400): rewrites the lit scene's pixels of see-through
// and mirror bodies (and the fluid filter's bridges between fluid bodies); every other pixel is discarded, so it
// keeps the composed colour byte for byte.
//
// Pseudo-3D: the viewer looks straight down; a body is a bump whose surface normal is (bevel n.xy, sqrt(1 - |n|^2)).
//   refraction  what is behind the body, displaced by the bent view ray: sample "behind" at
//               p - n.xy / n.z * (1 - 1 / ior) * (REFRACT_DEPTH_PX + thickness / 2), at least MIN_REFRACT_PX on
//               the tilted rim, attenuated by Beer-Lambert exp(-absorption * thickness) toward the body's colour
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

// What a ray sees at q in the plane: the lit scene; under a see-through body the floor; on the body `self_id` it
// was reflected from (a flat mirror / glass top reflecting its own area), the blurred surroundings (inpainted),
// blending into the scene just outside the nearest edge over MIRROR_SOFT_PX (no seam where rays start leaving it).
vec3 env(vec2 q, uint self_id) {
    ivec2 c = pix(q);
    uint id = mat_id(c);
    if (id != 0u && id == self_id) {
        float de = edge_d(c);
        vec3 sharp = texelFetch(u_scene, pix(q + edge_normal(c, vec2(0.0, -1.0)) * (de + 1.0)), 0).rgb;
        return mix(texture(u_fill_col, fill_uv(c)).rgb, sharp, exp(-de / MIRROR_SOFT_PX));
    }
    if (!see_through(kind_of(id))) return texelFetch(u_scene, c, 0).rgb;
    if (u_behind_on) return max((texelFetch(u_behind, c, 0).rgb + RATIO_EPS) * light_ratio(c) - RATIO_EPS, 0.0);
    return texture(u_fill_col, fill_uv(c)).rgb;
}

vec3 normal3(vec2 n) {
    float l2 = dot(n, n);
    if (l2 > 0.97) { n *= sqrt(0.97 / l2); l2 = 0.97; }
    return vec3(n, sqrt(1.0 - l2));
}

vec3 reflection(ivec2 p, vec3 n3, uint self_id, float d) {
    vec3 R = reflect(VIEW, n3);
    float l = length(R.xy);
    vec2 start = vec2(p) + 0.5;
    float nl = length(n3.xy);
    if (nl > 0.05) start += n3.xy / nl * d;  // the nearest edge point: the rim reflects what's beside it
    if (l < 1e-4) return env(start, self_id);
    vec2 dir = R.xy / l;
    float travel = min(MIRROR_DEPTH_PX * l / max(R.z, 0.05), MARCH_MAX_T);
#ifdef USE_SDF
    Ray r = march(start, dir, travel, self_id);
    vec3 c = r.end == 1 ? texelFetch(u_scene, pix(r.pos), 0).rgb : env(r.pos, r.events == 0 ? self_id : 0u);
    return c * r.thr;
#else
    return env(start + dir * travel, self_id);
#endif
}

vec3 glint(vec3 n3, float specular, float roughness) {
    vec3 h = normalize(normalize(GLINT_DIR) + vec3(0.0, 0.0, 1.0));
    float shin = exp2(10.0 * (1.0 - roughness));
    return vec3(GLINT_STRENGTH * specular * pow(max(dot(n3, h), 0.0), shin));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p.x >= u_clip.x || p.y >= u_clip.y) discard;
    uint id = mat_id(p);
    int kind = kind_of(id);
    Mat m = g_mat;
    vec4 nb = texelFetch(u_normal, p, 0);
    vec2 n = nb.rg * 2.0 - 1.0;
    float d = nb.b * 255.0;
    vec3 own = texelFetch(u_scene, p, 0).rgb;
    vec3 body = texelFetch(u_albedo, p, 0).rgb;  // the colour absorbed toward
    float thick = THICK_PER_EDGE * d;

    if (u_fluid_on && (kind == KIND_FLUID || kind < 0)) {
        vec2 pc = vec2(p) + 0.5;
        float f = half_at(u_field, pc).r;
        if (f < u_fluid_t) {
            if (kind == KIND_FLUID) { f_color = vec4(env(vec2(p) + 0.5, 0u), 1.0); return; }  // filtered away
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
        vec3 refr = env(vec2(p) + 0.5 + off, 0u) * mix(body, vec3(1.0), exp(-m.absorption * thick));
        vec3 col = mix(own, refr, m.transparency);
        float f0 = max(m.reflectivity, pow((ior - 1.0) / (ior + 1.0), 2.0));
        float F = f0 + (1.0 - f0) * pow(1.0 - n3.z, 5.0);
        if (F > 0.01) col = mix(col, reflection(p, n3, id, d), F);
        if (u_spec_on) col += texelFetch(u_spec, p, 0).rgb * m.transparency * (1.0 - F);
        col += glint(n3, m.specular, m.roughness);
        f_color = vec4(col, 1.0);
        return;
    }
    if (kind == KIND_MIRROR) {
        vec3 n3 = normal3(n);
        vec3 refl = reflection(p, n3, id, d) * mix(vec3(1.0), texelFetch(u_albedo, p, 0).rgb, m.tint_strength);
        vec3 col = mix(own, refl, m.reflectivity);
        if (u_spec_on) col += texelFetch(u_spec, p, 0).rgb * m.reflectivity;
        col += glint(n3, m.specular, m.roughness);
        f_color = vec4(col, 1.0);
        return;
    }
    discard;
}
