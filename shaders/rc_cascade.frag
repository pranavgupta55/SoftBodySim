// Radiance cascades, one cascade (gl_sdf.py GiEffect), after jason.today/rc: run for cascade n-1 down to 0, each
// merging the one above it. All cascades share one texture size E (the half-res GI grid padded to a multiple of
// 2^(n-1)) in a direction-first layout:
//   cascade i: probes every 2^i texels ((E / 2^i) per axis), 4^(i+1) rays per probe in 4^i groups of 4. The texture
//   is a 2^i x 2^i grid of tiles, tile g = the probes' group g; a texel holds its probe's mean over the group's 4
//   rays (pre-averaging): rgb radiance, a visibility (1 = nothing hit, less through see-through bodies).
//   Ray j's angle is (j + 0.5) * TAU / 4^(i+1); its 4 sub-rays in cascade i+1 are 4j..4j+3 = cascade i+1's group j.
//   Ray interval: [l0 (4^i - 1) / 3, l0 (4^(i+1) - 1) / 3] E texels from the probe (each 4x longer than the last).
// A ray that reaches its interval's end unblocked takes the radiance cascade i+1 saw in its direction, bilinearly
// interpolated between the 4 nearest cascade i+1 probes. Cascade 0's tile is the whole texture: per-texel fluence.
// Needs sdf_common.glsl.
uniform sampler2D u_scene;   // E: rgb emission, a: 1 opaque (rc_scene.frag)
uniform sampler2D u_dist;    // E: signed distance to opaque texels, in texels (NEAREST)
uniform sampler2D u_sigma;   // E, mipmapped: the see-through medium's extinction per texel (rc_scene.frag)
uniform sampler2D u_glow;    // E, mipmapped: its emission per texel
uniform sampler2D u_upper;   // cascade i+1, when u_has_upper
uniform bool u_has_upper;
uniform int u_index;         // i
uniform float u_l0;          // cascade 0's interval length, texels
uniform vec2 u_period;       // texels, (0, 0) none
uniform float u_lod_cap;     // this cascade's cap on the medium's mip level (GiEffect: GI_GLASS_LOD_CAPS)
uniform float u_wrap0;       // on a periodic axis medium lookups wrap into [u_wrap0, period + u_wrap0)

out vec4 f_rad;

const float TAU = 6.28318530718;
const int RC_STEPS = 32;
// a ray dimmed below this by the medium ends there (nothing more from it or the cascade above: at most 2% of
// what it would still bring; 0.001 cost +0.75 ms on halloween for at most 1/255)
const float RC_EXIT = 0.02;
const float SIGMA_EPS = 1e-5;  // less extinction than this per texel: clear (and no emission: rc_scene)

// Point p (texels) as a texture coordinate of the medium, wrapped on periodic axes.
vec2 medium_uv(vec2 p, ivec2 size) {
    for (int a = 0; a < 2; a++) {
        if (u_period[a] > 0.0) p[a] -= u_period[a] * floor((p[a] - u_wrap0) / u_period[a]);
    }
    return p / vec2(size);
}

// The step [t, t + s] of a ray, its midpoint at p (texels), through the medium at mip level lod: the emission
// gathered (x what got through so far) and the visibility left. Uniform extinction sigma along it: x exp(-sigma s),
// and the emission e integrates to e (1 - exp(-sigma s)) / sigma.
void absorb(vec2 p, float lod, float s, ivec2 size, inout vec3 rad, inout float vis) {
    vec2 uv = medium_uv(p, size);
    float sigma = textureLod(u_sigma, uv, lod).r;
    if (sigma < SIGMA_EPS) return;
    float T = exp(-sigma * s);
    rad += vis * textureLod(u_glow, uv, lod).rgb * (1.0 - T) / sigma;
    vis *= T;
}

// rgb: the radiance gathered, a: what still gets through (0 = blocked). Opaque texels (u_dist < 0) stop the ray
// with their emission. In between it steps by the distance to them, through the see-through medium: each step reads
// its mean extinction and emission at the step's midpoint, at the mip level of the step's length capped at
// u_lod_cap. A step longer than that level's reach (2^(cap + 1)) first looks over its whole length at a coarse level:
// clear there, it goes all the way; any medium, and it is cut to 2^(cap + 1) (so a thin glass wall between two far
// apart midpoints isn't stepped over). A ray that runs out of steps before its interval's end crosses the rest in
// one coarse look (no opaque check there): the medium past the last step is never skipped.
vec4 march(vec2 o, vec2 dir, float a, float b, ivec2 size) {
    float t = a;
    vec3 rad = vec3(0.0);
    float vis = 1.0;
    float cut = exp2(u_lod_cap + 1.0);
    for (int k = 0; k < RC_STEPS && t < b; k++) {
        ivec2 q = ivec2(floor(o + dir * t));
        if (!sdf_wrap(q, size, u_period)) return vec4(rad, vis);  // off the map: nothing there
        float d = texelFetch(u_dist, q, 0).r;
        if (d < 0.0) return vec4(rad + vis * texelFetch(u_scene, q, 0).rgb, 0.0);  // opaque / an emitter
        float s = min(max(d - 0.5, 0.5), b - t);
        if (s > cut) {
            if (textureLod(u_sigma, medium_uv(o + dir * (t + 0.5 * s), size), log2(s) + 0.5).r <= 0.0) {
                t += s;
                continue;
            }
            s = cut;
        }
        absorb(o + dir * (t + 0.5 * s), min(log2(s), u_lod_cap), s, size, rad, vis);
        t += s;
        if (vis < RC_EXIT) return vec4(rad, 0.0);
    }
    if (t < b) absorb(o + dir * (0.5 * (t + b)), log2(b - t), b - t, size, rad, vis);
    return vec4(rad, vis);
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
