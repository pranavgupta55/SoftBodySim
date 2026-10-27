// Lights pass (gl_lights.py): one light's diffuse and specular at this pixel, blended additively into
// shared["light"] (location 0) and shared["spec"] (location 1).
//
// Diffuse: N.L with N from the G-buffer normal (z implied) and L towards the light raised to its pseudo height, so
// bevelled edges facing the light brighten; with u_relief > 0 measured relative to a flat pixel (which gets 1).
// Specular: Blinn-Phong, view straight down (0, 0, 1), shininess = exp2(10 * (1 - roughness)) (roughness clamped
// to [0, 1]), strength material.specular, scaled by min(1, (shin + 2) / (SPEC_REF_SHIN + 2)) so a broad lobe is a
// soft sheen instead of a white flood. Toon materials (bands > 0) quantise both. Unlit pixels (flag bit2 clear) get
// nothing. Each light's term is multiplied by its shadow (sdf_shadow_pt / sdf_shadow_dir, or the stub's
// sdf_shadow when there is no SDF) unless it doesn't cast shadows. On periodic maps a point / spot light adds every
// image of itself within its radius (at most MAX_IMAGES, nearest first), each with its own attenuation and march.
//
// Included before it: material.glsl (Mat, mat_fetch) and sdf_shadow.glsl, or light_noshadow.glsl without an SDF.
layout(std140) uniform Lights {
    vec4 u_lights[1024];
};

uniform sampler2D u_normal;     // G-buffer normal: rg = n.xy * 0.5 + 0.5, a = flags / 255
uniform usampler2D u_mat;       // G-buffer material id
uniform sampler2D u_mat_table;  // material table (mat_fetch)
uniform sampler2D u_sdf;        // shared["sdf"] (unused by the no-SDF stub)
uniform vec2 u_period;
uniform float u_shadow_reach;   // directional lights: how far the shadow march looks, px
uniform float u_shadow_fade_start;  // directional lights: the shadow fades out from here to u_shadow_reach (<= 0 off)
uniform float u_relief;         // gl_lights.DIFFUSE_RELIEF: 0 = plain N.L, > 0 = flat 1 + relief x (N.L - flat N.L)

flat in int v_light;

layout(location = 0) out vec4 f_light;
layout(location = 1) out vec4 f_spec;

const float SPEC_REF_SHIN = 32.0;  // lobes at least this tight keep full strength (glass_resolve's glint() too)
const int MAX_IMAGES = 4;          // periodic light images per pixel (the nearest ones)
const float IMAGE_MIN = 1.0 / 255.0;  // a further image fainter than this is skipped
// the light's images around the minimum-image one (offsets in periods)
const vec2 IMAGE_ORDER[9] = vec2[9](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
                                    vec2(1.0, 1.0), vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(-1.0, -1.0));

vec2 min_image(vec2 d) {
    for (int k = 0; k < 2; k++) {
        if (u_period[k] > 0.0) {
            d[k] -= u_period[k] * floor(d[k] / u_period[k] + 0.5);
        }
    }
    return d;
}

float toon(float x, float bands) {
    return bands > 0.0 ? floor(clamp(x, 0.0, 1.0) * bands + 0.5) / bands : x;
}

// One call site for the march (it inlines: several sites cost registers and occupancy).
float shadow(vec2 p, vec2 dir, float reach, float soft, bool point) {
#ifdef SDF_SHADOW_GLSL
    return sdf_shadow_light(u_sdf, p, dir, reach, soft, u_period, point, point ? 0.0 : u_shadow_fade_start);
#else
    return sdf_shadow(u_sdf, p, dir, reach, soft, u_period);
#endif
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 nrm = texelFetch(u_normal, px, 0);
    uint flags = uint(nrm.a * 255.0 + 0.5);
    if ((flags & 4u) == 0u) {
        discard;
    }
    vec2 nxy = nrm.rg * 2.0 - 1.0;
    vec3 n = vec3(nxy, sqrt(max(1.0 - dot(nxy, nxy), 0.0)));

    vec4 a = u_lights[4 * v_light];
    vec4 b = u_lights[4 * v_light + 1];
    vec4 c = u_lights[4 * v_light + 2];
    vec4 e = u_lights[4 * v_light + 3];  // falloff, softness px, casts shadows, unused
    int kind = int(a.w + 0.5);
    vec2 p = gl_FragCoord.xy;

    Mat m = mat_fetch(u_mat_table, texelFetch(u_mat, px, 0).r);
    float bands = float(m.bands);
    float shin = exp2(10.0 * (1.0 - clamp(m.roughness, 0.0, 1.0)));
    float spec_norm = min(1.0, (shin + 2.0) / (SPEC_REF_SHIN + 2.0));

    // directional: one pass of the loop; point / spot: the light's images within its radius
    vec2 d0 = kind == 2 ? vec2(0.0) : min_image(a.xy - p);
    int nx = (kind != 2 && u_period.x > 0.0) ? 1 : 0;
    int ny = (kind != 2 && u_period.y > 0.0) ? 1 : 0;
    float diff = 0.0, spec = 0.0;
    int used = 0;  // bit j: IMAGE_ORDER[j] done (or not an image on this map)
    for (int j = 0; j < 9; j++) {
        ivec2 oj = ivec2(IMAGE_ORDER[j]);
        if (abs(oj.x) > nx || abs(oj.y) > ny) {
            used |= 1 << j;
        }
    }
    int n_images = kind == 2 ? 1 : MAX_IMAGES;  // (a uniform-dependent bound: the loop stays a loop)
    for (int it = 0; it < n_images; it++) {
        // the nearest image not yet done (directional: the one pass)
        int jbest = -1;
        float best = 1e30;
        for (int j = 0; j < 9; j++) {
            float dj = length(d0 + IMAGE_ORDER[j] * u_period);
            if ((used & (1 << j)) == 0 && dj < best) {
                best = dj;
                jbest = j;
            }
        }
        if (jbest < 0) {
            break;
        }
        used |= 1 << jbest;
        ivec2 oi = ivec2(IMAGE_ORDER[jbest]);
        float atten = 1.0;
        vec3 l;
        vec2 to_light;  // unit, in the plane: the shadow march direction
        float reach;
        if (kind == 2) {
            to_light = -c.xy;
            l = normalize(vec3(to_light, a.z));  // a.z: the elevation's tangent for directional lights
            reach = u_shadow_reach;
        } else {
            vec2 d = d0 + vec2(oi) * u_period;
            float dist = length(d);
            if (dist >= b.w) {
                break;  // the nearest left is out of reach: so are the rest
            }
            atten = pow(1.0 - dist / b.w, e.x);
            if (kind == 1) {
                float cosang = dist > 0.0 ? dot(-d / dist, c.xy) : 1.0;
                atten *= smoothstep(c.w, c.z, cosang);
            }
            // a further image too faint to show (under 1/255 before its shadow) isn't worth its march
            if (atten <= 0.0 || (it > 0 && atten * max(b.r, max(b.g, b.b)) < IMAGE_MIN)) {
                continue;
            }
            l = normalize(vec3(d, a.z));
            to_light = dist > 0.0 ? d / dist : vec2(0.0, -1.0);
            reach = dist;
        }
        if (e.z > 0.5) {
            atten *= shadow(p, to_light, reach, e.y, kind != 2);
        }
        if (atten <= 0.0) {
            continue;
        }
        float ndl = u_relief > 0.0 ? max(1.0 + u_relief * (dot(n, l) - l.z), 0.0) : max(dot(n, l), 0.0);
        diff += toon(ndl * atten, bands);
        if (m.specular > 0.0 && dot(n, l) >= 0.0) {
            vec3 h = normalize(l + vec3(0.0, 0.0, 1.0));
            spec += toon(pow(max(dot(n, h), 0.0), shin) * atten, bands) * m.specular * spec_norm;
        }
    }
    if (diff <= 0.0 && spec <= 0.0) {
        discard;
    }
    f_light = vec4(b.rgb * diff, 1.0);
    f_spec = vec4(b.rgb * spec, 1.0);
}
