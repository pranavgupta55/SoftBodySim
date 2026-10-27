// Radiance cascades, scene pass (gl_sdf.py GiEffect): the G-buffer at half resolution as the GI scene. Each texel
// covers a 2x2 block of render px: rgb = mean emission of the block's solid px, a = 1 if any px is solid.
// Solid = a shadow-casting occluder (blocks light) or an emissive body px (gives light). Emission is the px's albedo
// x material emissive x emissive_strength. Needs material.glsl (mat_fetch) and sdf_common.glsl.
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
    for (int k = 0; k < 4; k++) {
        ivec2 p = 2 * q + ivec2(k & 1, k >> 1);
        if (any(greaterThanEqual(p, size))) continue;
        bool solid = sdf_solid(u_normal, p, 0);
        vec3 pe = vec3(0.0);
        uint id = texelFetch(u_mat, p, 0).r;
        if (id != 0u) {
            Mat m = mat_fetch(u_mats, id);
            pe = texelFetch(u_albedo, p, 0).rgb * m.emissive * m.emissive_strength;
        }
        if (solid || max(pe.r, max(pe.g, pe.b)) > 0.0) {
            e += pe;
            n += 1.0;
        }
    }
    f_scene = n > 0.0 ? vec4(e / n * u_emit_gain, 1.0) : vec4(0.0);
}
