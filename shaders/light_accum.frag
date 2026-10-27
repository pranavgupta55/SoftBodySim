// Lights pass (gl_lights.py): one light's diffuse and specular at this pixel, blended additively into
// shared["light"] (location 0) and shared["spec"] (location 1).
//
// Diffuse: N.L with N from the G-buffer normal (z implied) and L towards the light raised to its pseudo height, so
// bevelled edges facing the light brighten. Specular: Blinn-Phong, view straight down (0, 0, 1), shininess =
// exp2(10 * (1 - roughness)), strength material.specular. Toon materials (bands > 0) quantise both. Unlit pixels
// (flag bit2 clear) get nothing. Each light's term is multiplied by its shadow (sdf_shadow, or the stub when there
// is no SDF) unless it doesn't cast shadows. Distances are minimum-image across u_period.
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

flat in int v_light;

layout(location = 0) out vec4 f_light;
layout(location = 1) out vec4 f_spec;

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

    float atten = 1.0;
    vec3 l;
    vec2 to_light;  // unit, in the plane: the shadow march direction
    float reach;
    if (kind == 2) {
        to_light = -c.xy;
        l = normalize(vec3(to_light, a.z));  // a.z: the elevation's tangent for directional lights
        reach = u_shadow_reach;
    } else {
        vec2 d = min_image(a.xy - p);
        float dist = length(d);
        if (dist >= b.w) {
            discard;
        }
        atten = pow(1.0 - dist / b.w, e.x);
        if (kind == 1) {
            float cosang = dist > 0.0 ? dot(-d / dist, c.xy) : 1.0;
            atten *= smoothstep(c.w, c.z, cosang);
        }
        if (atten <= 0.0) {
            discard;
        }
        to_light = dist > 0.0 ? d / dist : vec2(0.0, -1.0);
        l = normalize(vec3(d, a.z));
        reach = dist;
    }
    if (e.z > 0.5) {
        atten *= sdf_shadow(u_sdf, p, to_light, reach, e.y, u_period);
        if (atten <= 0.0) {
            discard;
        }
    }

    Mat m = mat_fetch(u_mat_table, texelFetch(u_mat, px, 0).r);
    float bands = float(m.bands);
    float diff = toon(max(dot(n, l), 0.0) * atten, bands);
    float spec = 0.0;
    if (m.specular > 0.0) {
        vec3 h = normalize(l + vec3(0.0, 0.0, 1.0));
        float shin = exp2(10.0 * (1.0 - m.roughness));
        spec = toon(pow(max(dot(n, h), 0.0), shin) * atten, bands) * m.specular;
        spec *= step(0.0, dot(n, l));
    }
    f_light = vec4(b.rgb * diff, 1.0);
    f_spec = vec4(b.rgb * spec, 1.0);
}
