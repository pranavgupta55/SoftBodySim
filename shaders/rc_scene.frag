// Radiance cascades, scene pass (gl_sdf.py GiEffect): the G-buffer at half resolution as the GI scene. Each texel
// covers a 2x2 block of render px: rgb = mean emission of the block's solid px, a = 0 open, 1 solid (blocks light),
// 1 + T for a block of translucent px only: a see-through body (FLAG_TRANSLUCENT) that lets the mean T of its px's
// transmittance (material.glsl mat_transmittance, luminance) through (rc_cascade goes on through it).
// Solid = a shadow-casting occluder (blocks light) or an emissive body px (gives light). Emission is the px's albedo
// x material emissive x emissive_strength, 0 for an unlit material (compose shows it unlit: no glow there, so
// none thrown either). Needs material.glsl (mat_fetch) and sdf_common.glsl.
uniform sampler2D u_normal;   // G-buffer normal (flags in alpha)
uniform sampler2D u_albedo;   // G-buffer albedo
uniform usampler2D u_mat;     // G-buffer material id
uniform sampler2D u_mats;     // material table
uniform float u_emit_gain;

out vec4 f_scene;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_normal, 0);
    vec3 e = vec3(0.0);
    float n = 0.0;
    float tsum = 0.0;  // the translucent px' transmittance
    bool opaque = false;  // a solid px that isn't translucent
    for (int k = 0; k < 4; k++) {
        ivec2 p = 2 * q + ivec2(k & 1, k >> 1);
        if (any(greaterThanEqual(p, size))) continue;
        bool solid = sdf_solid(u_normal, p, 0);
        vec3 pe = vec3(0.0);
        uint id = texelFetch(u_mat, p, 0).r;
        Mat m;
        if (id != 0u) {
            m = mat_fetch(u_mats, id);
            pe = texelFetch(u_albedo, p, 0).rgb * m.emissive * m.emissive_strength * float(m.lit);  // unlit: no glow
        }
        if (solid || max(pe.r, max(pe.g, pe.b)) > 0.0) {
            e += pe;
            n += 1.0;
            int f = int(texelFetch(u_normal, p, 0).a * 255.0 + 0.5);
            if (solid && (f & FLAG_TRANSLUCENT) != 0 && id != 0u && mat_translucent(m)) {
                // (a translucent body's black border passes light untinted, as in its shadow: sdf_tr.frag)
                vec3 col = (f & FLAG_BORDER) != 0 ? vec3(1.0) : texelFetch(u_albedo, p, 0).rgb;
                tsum += dot(mat_transmittance(m, col), vec3(0.2126, 0.7152, 0.0722));
            } else {
                opaque = true;
            }
        }
    }
    float a = (n > 0.0 && !opaque) ? 1.0 + clamp(tsum / n, 0.0, 1.0) : 1.0;
    f_scene = n > 0.0 ? vec4(e / n * u_emit_gain, a) : vec4(0.0);
}
