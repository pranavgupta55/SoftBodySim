// Caustics (gl_glass.CausticsEffect): one point per (ray, sample). Vertex k = ray k / u_samples, sample k % u_samples.
// Each light shoots u_rays rays in the plane (point: all around; spot: across its cone; directional: parallel, across
// the view); a sample marches its ray (glass_common.march: Snell, TIR, mirrors) for a jittered path length t and, if
// the ray has been refracted or reflected on the way and is in open air there, deposits light at that point. Point
// density along converging rays is what focuses into caustics; the weight makes evenly spread rays give the light's
// own unshadowed irradiance (rays t * dTheta apart, u_samples / u_max_t samples per px of path).
uniform int u_kind;          // 0 point, 1 spot, 2 directional
uniform vec2 u_light_pos;    // px
uniform vec2 u_light_dir;
uniform float u_cone;        // spot half angle, radians
uniform vec3 u_light_col;    // colour x intensity x CAUSTIC_GAIN
uniform float u_radius_px;
uniform float u_falloff;
uniform int u_rays;
uniform int u_samples;
uniform float u_max_t;
uniform vec2 u_size;

out vec3 v_col;

float hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return float(x) / 4294967296.0;
}

void main() {
    int ray = gl_VertexID / u_samples, j = gl_VertexID - ray * u_samples;
    float u = (float(ray) + 0.5) / float(u_rays);
    float diag = length(vec2(u_clip));
    vec2 o, dir;
    float spacing;  // px between neighbouring rays at path length t (per unit t for point / spot)
    if (u_kind == 2) {
        dir = normalize(u_light_dir);
        vec2 perp = vec2(-dir.y, dir.x);
        o = 0.5 * vec2(u_clip) - dir * 0.5 * diag + perp * (u - 0.5) * diag;
        spacing = diag / float(u_rays);
    } else {
        float a0 = atan(u_light_dir.y, u_light_dir.x);
        float a = u_kind == 1 ? a0 + (2.0 * u - 1.0) * u_cone : 6.2831853 * u;
        dir = vec2(cos(a), sin(a));
        o = u_light_pos;
        spacing = (u_kind == 1 ? 2.0 * u_cone : 6.2831853) / float(u_rays);
    }
    float t = (float(j) + hash(uint(gl_VertexID) * 747796405u + 2891336453u)) / float(u_samples) * u_max_t;
    Ray r = march(o, dir, t, 0u);
    v_col = vec3(0.0);
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);  // off screen: nothing deposited
    if (r.end != 0 || r.events == 0 || off_view(r.pos)) return;
    float fall = u_kind == 2 ? 1.0 : pow(max(1.0 - t / u_radius_px, 0.0), u_falloff);
    float w = (u_kind == 2 ? spacing : spacing * max(t, 1.0)) * u_max_t / float(u_samples);
    v_col = u_light_col * r.thr * fall * w;
    vec2 q = vec2(pix(r.pos)) + 0.5;
    gl_Position = vec4(q / u_size * 2.0 - 1.0, 0.0, 1.0);
}
