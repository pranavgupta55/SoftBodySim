// Radiance cascades, one cascade (gl_sdf.py GiEffect), after jason.today/rc: run for cascade n-1 down to 0, each
// merging the one above it. All cascades share one texture size E (the half-res GI grid padded to a multiple of
// 2^(n-1)) in a direction-first layout:
//   cascade i: probes every 2^i texels ((E / 2^i) per axis), 4^(i+1) rays per probe in 4^i groups of 4. The texture
//   is a 2^i x 2^i grid of tiles, tile g = the probes' group g; a texel holds its probe's mean over the group's 4
//   rays (pre-averaging): rgb radiance, a visibility (1 = nothing hit).
//   Ray j's angle is (j + 0.5) * TAU / 4^(i+1); its 4 sub-rays in cascade i+1 are 4j..4j+3 = cascade i+1's group j.
//   Ray interval: [l0 (4^i - 1) / 3, l0 (4^(i+1) - 1) / 3] E texels from the probe (each 4x longer than the last).
// A ray that reaches its interval's end unblocked takes the radiance cascade i+1 saw in its direction, bilinearly
// interpolated between the 4 nearest cascade i+1 probes. Cascade 0's tile is the whole texture: per-texel fluence.
// Needs sdf_common.glsl.
uniform sampler2D u_scene;   // E: rgb emission, a solid (rc_scene.frag)
uniform sampler2D u_dist;    // E: signed distance to solid texels, in texels (NEAREST)
uniform sampler2D u_upper;   // cascade i+1, when u_has_upper
uniform bool u_has_upper;
uniform int u_index;         // i
uniform float u_l0;          // cascade 0's interval length, texels
uniform vec2 u_period;       // texels, (0, 0) none

out vec4 f_rad;

const float TAU = 6.28318530718;
const int RC_STEPS = 32;

vec4 march(vec2 o, vec2 dir, float a, float b, ivec2 size) {
    float t = a;
    for (int k = 0; k < RC_STEPS && t <= b; k++) {
        ivec2 q = ivec2(floor(o + dir * t));
        if (!sdf_wrap(q, size, u_period)) return vec4(0.0, 0.0, 0.0, 1.0);  // off the map: nothing there
        float d = texelFetch(u_dist, q, 0).r;
        if (d < 0.0) return vec4(texelFetch(u_scene, q, 0).rgb, 0.0);
        t += max(d - 0.5, 0.5);
    }
    return vec4(0.0, 0.0, 0.0, 1.0);
}

// Cascade i+1's radiance along its group j (= this cascade's ray j) at probe-centre position o (texels).
// On a periodic axis a 2x2 footprint that would leave the probe grid is taken one period over instead (the grid is
// padded past the period for this, GiEffect._alloc; probes past the period see the wrapped world): no seam.
vec4 upper(int j, vec2 o, ivec2 size) {
    int n2 = 2 << u_index;
    ivec2 tsize = size / n2;
    ivec2 origin = ivec2(j % n2, j / n2) * tsize;
    vec2 u = o / float(n2) - 0.5;
    for (int a = 0; a < 2; a++) {
        if (u_period[a] > 0.0) {
            if (u[a] < 0.0) u[a] += u_period[a] / float(n2);
            else if (u[a] + 1.0 > float(tsize[a] - 1)) u[a] -= u_period[a] / float(n2);
        }
    }
    vec2 i0 = floor(u);
    vec2 f = u - i0;
    ivec2 b = ivec2(i0);
    ivec2 hi = tsize - 1;
    vec4 c00 = texelFetch(u_upper, origin + clamp(b, ivec2(0), hi), 0);
    vec4 c10 = texelFetch(u_upper, origin + clamp(b + ivec2(1, 0), ivec2(0), hi), 0);
    vec4 c01 = texelFetch(u_upper, origin + clamp(b + ivec2(0, 1), ivec2(0), hi), 0);
    vec4 c11 = texelFetch(u_upper, origin + clamp(b + ivec2(1, 1), ivec2(0), hi), 0);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_scene, 0);
    int n = 1 << u_index;
    ivec2 tsize = size / n;
    ivec2 tile = c / tsize;
    ivec2 q = c - tile * tsize;
    int g = tile.y * n + tile.x;
    float nrays = float(4 << (2 * u_index));
    vec2 o = (vec2(q) + 0.5) * float(n);
    float p4 = float(1 << (2 * u_index));
    float a = u_l0 * (p4 - 1.0) / 3.0;
    float b = u_l0 * (4.0 * p4 - 1.0) / 3.0;
    vec4 sum = vec4(0.0);
    for (int k = 0; k < 4; k++) {
        int j = 4 * g + k;
        float ang = (float(j) + 0.5) * TAU / nrays;
        vec4 r = march(o, vec2(cos(ang), sin(ang)), a, b, size);
        if (u_has_upper && r.a > 0.0) {
            vec4 up = upper(j, o, size);
            r = vec4(r.rgb + r.a * up.rgb, r.a * up.a);
        }
        sum += r;
    }
    f_rad = sum * 0.25;
}
