// Radiance cascades, scene pass (gl_sdf.py GiEffect): the G-buffer at half resolution as the GI scene, in two
// textures. Each texel covers a 2x2 block of render px.
//   location 0 (f_scene): rgb = mean emission of the block's solid px, a = 1 where the block has an opaque px (blocks
//     light; the GI's jump flood seeds from these), 0 otherwise.
//   locations 1 and 2 (f_sigma R16F, f_glow RGBA16F, both mipmapped by GiEffect): the see-through bodies
//     (FLAG_TRANSLUCENT) as a participating medium: sigma, the mean extinction per texel of path over the block's 4
//     px (open px 0), and their emission per texel of path. 0 in a block that has an opaque px (the ray stops there
//     anyway); no emission where sigma is 0 (rc_cascade skips such texels).
// A px of transmittance T (material.glsl mat_transmittance, luminance) gets sigma = -ln T / u_chord: a ray crossing
// one u_chord of it (a typical body's mean chord) keeps T, as when GI counted a factor T per body crossed; its
// emission E becomes E sigma / (1 - T) per texel, so that crossing gives E, as before (a fully clear px, T = 1,
// neither dims nor glows: there is no medium to emit).
// On a periodic axis, texels past the G-buffer (the grid runs on past the period, GiEffect._alloc) hold the medium
// one period back, so its mip levels are whole across the seam (rc_cascade samples them there).
// Points view: a see-through silhouette's transmittance comes from gbuf["tr"] and its emission from gbuf["tr_emit"]
// (gl_world _occluder_fill: its px have no material id of their own; a stick / dot on it emits the more of the two,
// so an emissive glass tile glows into the GI as in the filled view), and an emissive stick / dot of a see-through body outside the silhouette (no
// FLAG_OCCLUDER; its twin material keeps the body's transparency) is translucent like the body.
// Solid = a shadow-casting occluder (blocks light) or an emissive body px (gives light). Emission is the px's albedo
// x material emissive x emissive_strength, 0 for an unlit material (compose shows it unlit: no glow there, so
// none thrown either). A filled body's black border (FLAG_BORDER) is drawn black but emits like the body: the albedo
// of the nearest non-border px of the same material within BORDER_REACH px (the body's own fill just inside; a
// touching body of another material never matches; none found: no glow). Needs material.glsl and sdf_common.glsl.
uniform sampler2D u_normal;   // G-buffer normal (flags in alpha)
uniform sampler2D u_albedo;   // G-buffer albedo
uniform usampler2D u_mat;     // G-buffer material id
uniform sampler2D u_mats;     // material table
uniform float u_emit_gain;
uniform sampler2D u_tr;       // the points view's gbuf["tr"] (gl_world _occluder_fill), when u_has_tr
uniform bool u_has_tr;
uniform sampler2D u_tr_emit;  // the points view's gbuf["tr_emit"] (with gbuf["tr"]), when u_has_tr_emit
uniform bool u_has_tr_emit;
uniform float u_chord;        // the see-through bodies' reference chord, texels (GiEffect: GI_GLASS_CHORD)
uniform vec2 u_period;        // texels, (0, 0) none
const int BORDER_REACH = 3;  // > gl_world.BORDER_WIDTH (2): the fill starts just inside the 2px outline

// The albedo a border px at p of material id would have: the nearest non-border px of that material's.
vec3 border_albedo(ivec2 p, uint id, ivec2 size) {
    vec3 a = vec3(0.0);
    int best = BORDER_REACH * BORDER_REACH + 1;
    for (int dy = -BORDER_REACH; dy <= BORDER_REACH; dy++) {
        for (int dx = -BORDER_REACH; dx <= BORDER_REACH; dx++) {
            int d2 = dx * dx + dy * dy;
            ivec2 s = p + ivec2(dx, dy);
            if (d2 >= best || any(lessThan(s, ivec2(0))) || any(greaterThanEqual(s, size))) continue;
            uint f = uint(texelFetch(u_normal, s, 0).a * 255.0 + 0.5);
            if ((f & uint(FLAG_BORDER)) != 0u || texelFetch(u_mat, s, 0).r != id) continue;
            a = texelFetch(u_albedo, s, 0).rgb;
            best = d2;
        }
    }
    return a;
}

layout(location = 0) out vec4 f_scene;
layout(location = 1) out float f_sigma;
layout(location = 2) out vec4 f_glow;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_normal, 0);
    vec3 e = vec3(0.0);
    float n = 0.0;
    bool opaque = false;  // a solid px that isn't translucent
    vec4 glass = vec4(0.0);  // the see-through px' emission and extinction, summed
    // past the G-buffer on a periodic axis: the medium one period back (render px), the scene left empty
    ivec2 shift = ivec2(0);
    bool beyond = false;
    for (int a = 0; a < 2; a++) {
        if (2 * q[a] >= size[a]) {
            if (u_period[a] <= 0.0) { f_scene = vec4(0.0); f_sigma = 0.0; f_glow = vec4(0.0); return; }
            beyond = true;
            float L = 2.0 * u_period[a];
            shift[a] = int(floor(float(2 * q[a]) / L) * L + 0.5);
        }
    }
    for (int k = 0; k < 4; k++) {
        ivec2 p = 2 * q + ivec2(k & 1, k >> 1) - shift;
        if (any(greaterThanEqual(p, size)) || any(lessThan(p, ivec2(0)))) continue;
        bool solid = sdf_solid(u_normal, p, 0);
        vec3 pe = vec3(0.0);
        uint id = texelFetch(u_mat, p, 0).r;
        Mat m;
        if (id != 0u) {
            m = mat_fetch(u_mats, id);
            vec3 k = m.emissive * m.emissive_strength * float(m.lit);  // unlit: no glow
            if (max(k.r, max(k.g, k.b)) > 0.0) {
                uint f = uint(texelFetch(u_normal, p, 0).a * 255.0 + 0.5);
                pe = k * ((f & uint(FLAG_BORDER)) != 0u ? border_albedo(p, id, size) : texelFetch(u_albedo, p, 0).rgb);
            }
        }
        if (solid || max(pe.r, max(pe.g, pe.b)) > 0.0) {
            int f = int(texelFetch(u_normal, p, 0).a * 255.0 + 0.5);
            vec4 ptr = u_has_tr ? texelFetch(u_tr, p, 0) : vec4(0.0);
            float t = -1.0;  // the px's transmittance; < 0: opaque
            if (solid && (f & FLAG_TRANSLUCENT) != 0 && ptr.a > 0.5) {
                // the points view's see-through silhouette: its transmittance from gbuf["tr"], its glow from
                // gbuf["tr_emit"] (or a stick / dot's own there, if brighter)
                t = dot(ptr.rgb, LUMA);
                if (u_has_tr_emit) pe = max(pe, texelFetch(u_tr_emit, p, 0).rgb);
            } else if (solid && (f & FLAG_TRANSLUCENT) != 0 && id != 0u && mat_translucent(m)) {
                // (a translucent body's black border passes light untinted, as in its shadow: sdf_tr.frag)
                vec3 col = (f & FLAG_BORDER) != 0 ? vec3(1.0) : texelFetch(u_albedo, p, 0).rgb;
                t = dot(mat_transmittance(m, col), LUMA);
            } else if ((f & FLAG_OCCLUDER) == 0 && id != 0u && m.transparency > 0.0) {
                // the points view's stick / dot of a see-through body (its twin material keeps the transparency,
                // gl_world _materials) reaching past the body's silhouette: it lets light through like the body
                vec3 col = clamp(texelFetch(u_albedo, p, 0).rgb, 0.0, 1.0);
                vec3 tv = clamp(m.transparency, 0.0, 1.0) * mix(col, vec3(1.0),
                                                                exp(-max(m.absorption, 0.0) * MAT_SHADOW_THICK_PX));
                t = dot(tv, LUMA);
            }
            if (t < 0.0) {
                opaque = true;
                e += pe;
                n += 1.0;
            } else {
                t = clamp(t, 1e-4, 1.0);
                float sigma = -log(t) / u_chord;
                if (t < 0.9999) glass += vec4(pe * u_emit_gain * sigma / (1.0 - t), sigma);
            }
        }
    }
    f_scene = opaque && !beyond ? vec4(e / n * u_emit_gain, 1.0) : vec4(0.0);
    f_sigma = opaque ? 0.0 : glass.a * 0.25;
    f_glow = opaque ? vec4(0.0) : vec4(glass.rgb * 0.25, 1.0);
}
